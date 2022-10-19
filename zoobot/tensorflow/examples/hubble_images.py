import os
import logging
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tqdm import tqdm

from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess, custom_layers
from zoobot.tensorflow.predictions import predict_on_dataset
from zoobot.tensorflow.training import training_config, losses
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.datasets import gz_hubble

#Image parameters
initial_size = 300 
crop_size = int(initial_size * 0.75)
resize_size = 224  #Zoobot, as pretrained, expects 224x224 images
file_format = "png"

#Batch size
batch_size = 128

#Read in the dataset
df = pd.read_csv("/content/drive/MyDrive/MPE/2022_Ben_Aussel/Data/Hubble_COSMOS_labels_complete.csv")

#Add the id_str column that contains the path to each image
paths = ["/content/content/pngs_hubble_complete/cosmos_acs_" + str(image_id) + ".png" for image_id in df["ObjNo"]]
df["id_str"] = paths

print("Number of labelled galaxies:",len(df))

def check_for_images(paths,labels):
    """
        Check if images are present and remove non-existing images.
    """
    print("There are labels for",len(paths),"images.")
    print("Checking if the the files exist...")
    paths_checked, labels_checked = [], []
    for i in tqdm(range(len(paths))):
        file_exists = os.path.exists(paths[i])
        if file_exists:
            paths_checked.append(paths[i])
            labels_checked.append(labels[i])
    print("\nCheck over. There are actually",len(paths_checked),"images that can be used for training.")
    return paths_checked, labels_checked

def get_paths_and_labels(data,schema,trafo_answers,trafo_total):
    """
        Gets the paths of the images and their corresponding labels given by the 'schema'.

        Returns the paths as a list of strings and the labels as a list of dictionaries.
            The keys of the dict are given by the decision tree answers and the values
            are the vote fractions multiplied by the total number of participants, so the
            number of volunteers giving that specific answer.
    """
    paths, labels = [], []
    answers = schema.label_cols
    questions = list(schema.question_answer_pairs.keys())
    answer_indices = schema.question_index_groups
    for i in tqdm(range(len(data))):
        label_dict = {}
        for j in range(len(questions)):
            number_votes = data[trafo_total[questions[j]]].iloc[i]
            for k in range(answer_indices[j][0],answer_indices[j][1]+1):
                label_dict[answers[k]] = np.round(data[trafo_answers[answers[k]]].iloc[i]*number_votes)
        image_id = np.array(data["ObjNo"].iloc[i])
        paths.append("/content/content/pngs_hubble_complete/cosmos_acs_" + str(image_id) + ".png")
        labels.append(label_dict)
    return paths, labels

def prepare_dataset(data,schema,trafo_answers,trafo_total):
    """
        Prepares the train and validation datasets for the given schema.
    """
    paths, labels = get_paths_and_labels(data,schema,trafo_answers,trafo_total)
    print("Examples - First five labels and paths:")
    print(labels[:5])
    print(paths[:5])
    
    paths_real, labels_real = check_for_images(paths,labels)

    #Randomly divide into train and validation sets using sklearn:
    paths_train, paths_val, labels_train, labels_val= train_test_split(paths_real, labels_real, test_size=0.2, random_state=0)
    assert set(paths_train).intersection(set(paths_val)) == set()  # check there's no train/val overlap

    file_format = 'png'
    batch_size = 128  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100

    raw_train_dataset = image_datasets.get_image_dataset(
        paths_train,
        file_format=file_format,
        requested_img_size=initial_size,
        batch_size=batch_size,
        labels=labels_train
    )

    raw_val_dataset = image_datasets.get_image_dataset(
        paths_val,
        file_format=file_format,
        requested_img_size=initial_size,
        batch_size=batch_size,
        labels=labels_val
    )

    print("First batch of val dataset:")
    print(list(raw_val_dataset.take(1)))  

    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=schema.label_cols,  
        input_size=initial_size,
        normalise_from_uint8=True,  # divide by 255
        make_greyscale=True,  # take the mean over RGB channels
        permute_channels=False # swap channels around randomly (no need when making greyscale anwyay)
    )

    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)

    return train_dataset, val_dataset, paths_train, paths_val

def prepare_model_votes(schema):
    """
        Prepares the pretrained regression model for the finetuning.
    """
    #checkpoint_dir = '/content/zoobot/replicated_train_only_greyscale_tf'
    checkpoint_dir = '/content/drive/MyDrive/MPE/2022_Ben_Aussel/Zoobot/efficientnet_dr5_tensorflow_greyscale_catalog_debug'

    checkpoint_loc = os.path.join(checkpoint_dir, 'checkpoint')

    # get headless model (inc. augmentations)
    base_model = define_model.load_model(
        checkpoint_loc,  # loading pretrained model as above
        expect_partial=True,  # ignores some optimizer warnings
        include_top=False,  # do not include the head used for GZ DECaLS, this time - we will add our own head
        input_size=initial_size,  # the preprocessing above did not change size
        crop_size=crop_size,  # model augmentation layers apply a crop...
        resize_size=resize_size,  # ...and then apply a resize
        output_dim=None  # headless so no effect
    )

    base_model.trainable = False  # freeze the headless model (no training allowed)

    new_head = tf.keras.Sequential([
    layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(schema.label_cols), activation=lambda x: tf.nn.sigmoid(x) * 100. + 1.)
    ])

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(initial_size, initial_size, 1)),
        base_model,
        new_head
    ])

    return model

def train_on_vote_labels(model, schema, train_dataset, val_dataset, save_path, lr=None, epochs=1000, reduce_patience=10, patience=20):
    """
        Trains the model with the given datasets (train_dataset, val_dataset) and saves the results at the given path.
    """

    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    def loss(x, y): return multiquestion_loss(x, y) / batch_size

    extra_callbacks = []
    if lr is None:
        reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=reduce_patience, cooldown=5, verbose=0)
        extra_callbacks = [reduce_on_plateau]
        adam_optimizer = tf.keras.optimizers.Adam()
    else:
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        loss=loss,
        optimizer=adam_optimizer
    )
    model.summary()

    train_config = training_config.TrainConfig(
        log_dir=save_path,
        epochs=epochs,
        patience=patience  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )
    
    # acts inplace on model
    # saves best checkpoint to train_config.logdir / checkpoints
    # also updates model to be that best checkpoint
    training_config.train_estimator(
        model,
        train_config,  # e.g. how to train epochs, patience
        train_dataset,
        val_dataset,
        extra_callbacks=extra_callbacks
    )

    return model

def make_predictions(model,paths,schema,save_path,save_name="val_predictions.csv"):
    """
        Makes predictions on images given by the 'paths' list with the given schema and saves them.
    """
    paths_pred = list(paths) 
    raw_pred_dataset = image_datasets.get_image_dataset(
        paths_pred,
        file_format=file_format,
        requested_img_size=initial_size,
        batch_size=batch_size
    )

    ordered_paths = [x.numpy().decode('utf8') for batch in raw_pred_dataset for x in batch['id_str']]

    # must exactly match the preprocessing you used for training
    pred_config = preprocess.PreprocessingConfig(
        label_cols=[],  # image_datasets.get_image_dataset will put the labels arg under 'label' key for each batch
        input_size=initial_size,
        make_greyscale=True,
        normalise_from_uint8=True,
        permute_channels=False
    )

    pred_dataset = preprocess.preprocess_dataset(raw_pred_dataset, pred_config)

    save_loc = save_path+save_name
    predict_on_dataset.predict(pred_dataset, model, 1, schema.label_cols, save_loc)

def predict_on_dataset_reg(paths,model,save_path,full_dataset,schema,save_name="full_predictions.csv"):
    print("Predicting on {} galaxies...".format(len(paths)))
    make_predictions(model,paths,schema,save_path,save_name=save_name)
    print("--> Done.")
    predictions_frac = concentrations_to_vote_fractions(schema,save_path,save_name)
    predictions_with_labels = pd.merge(predictions_frac, full_dataset, on='id_str', how='inner')
    predictions_with_labels.to_csv(save_path+save_name)
    print("All results saved at {}".format(save_path+save_name))

def concentrations_to_vote_fractions(schema,save_path,save_name):
    """ 
        Transforms the given concentrations into vote fractions for the given
        binary schema and save them.
    """
    dataset_pred = pd.read_csv(save_path+save_name)

    answers = list(schema.label_cols)
    for i in range(len(answers)): answers[i] = answers[i] + "_pred"
    answer_indices = list(schema.question_index_groups)
    print("Calculating vote fractions for {} questions with {} answers in total...".format(len(answer_indices),len(answers)))

    print("--> For {} galaxies".format(len(dataset_pred)))

    for i in range(len(answer_indices)): #For all questions
        data_concentrations = [] #The concentrations for all answers (index 0 is answer 0, ...)
        for j in range(answer_indices[i][0],answer_indices[i][1]+1): #All answers for that question       
            data_votes = list(dataset_pred[answers[j]])
            #Transform from list to float ([3.425] --> 3.425)
            data_conc = []
            for k in range(len(data_votes)):
                if isinstance(data_votes[k],float):
                    data_conc.append(data_votes[k])
                else:
                    data_conc.append(json.loads(data_votes[k])[0])
            data_concentrations.append(data_conc)
        data_concentrations = np.array(data_concentrations)
        #Calculate fractions
        total_votes = data_concentrations.sum(axis=0)
        data_frac = data_concentrations/total_votes
            
        for j in range(answer_indices[i][0],answer_indices[i][1]+1):
            for k in range(len(data_frac[j-answer_indices[i][0]])):
                if np.isnan(data_frac[j-answer_indices[i][0],k]):
                    data_frac[j-answer_indices[i][0],k] = 0
            dataset_pred[answers[j]+"_frac"] = data_frac[j-answer_indices[i][0],:]

    return dataset_pred

def train_finetuning_loop(model,schema,train_dataset,val_dataset,save_path,loop_yes=True):

    print("--> Training...")
    model = train_on_vote_labels(model,schema,train_dataset,val_dataset,save_path,lr=0.001)
    layers = ['top', 'block7', 'block6', 'block5', 'block4','block3','block2']

    if loop_yes:
        for i in range(len(layers)-1):
            print("Unfreezing model layers:",layers[:i+1])
            print("--> Training...")
            utils.unfreeze_model(model, unfreeze_names=layers[:i+1])
            model = train_on_vote_labels(model,schema,train_dataset,val_dataset,save_path,lr=0.0001)
        
    print("Unfreezing all model layers.")
    print("--> Training...")
    utils.unfreeze_model(model, unfreeze_names=[], unfreeze_all=True)
    model = train_on_vote_labels(model,schema,train_dataset,val_dataset,save_path,lr=0.0001)

    return model

def get_asked_questions(data_row,schema):
    questions_asked, given_answers, given_answers_frac = [], [], []

    #Invert the depenendencies dictionary to get the next question (following the decision tree)
    inv_dep = {}
    for k, v in schema.dependencies.items():
        inv_dep[v] = inv_dep.get(v, []) + [k]

    previous_answers = [None] #Start with no previous answer
    finished = False
    while not finished: #For all tier questions
        current_questions = []
        #Get all following questions that are asked
        for i in range(len(previous_answers)):
            if previous_answers[i] in inv_dep:
                for j in range(len(inv_dep[previous_answers[i]])):
                    question = inv_dep[previous_answers[i]][j]
                    current_questions.append(question)
                    questions_asked.append(question)
        #If there are no new asked questions stop the loop
        if current_questions == []:
            finished = True

        previous_answers = []
        #Get the answers with the maximum fraction
        for i in range(len(current_questions)):
            question = current_questions[i]
            answers = schema.question_answer_pairs.get(question) #All possible answers for that question
            answers_pred_frac = []
            for j in range(len(answers)):
                answers_pred_frac.append(data_row[question+answers[j]+"_pred_frac"]) #Predicted fractions
            index_max = np.argmax(answers_pred_frac) #Get the maximum fraction
            frac_max = np.max(answers_pred_frac)
            given_answer = question+answers[index_max] #Save the corresponding answer
            given_answers.append(given_answer)
            given_answers_frac.append(frac_max)
            previous_answers.append(given_answer)

    return questions_asked, given_answers, given_answers_frac

def mean_deviation_half(schema,trafo_answers,trafo_total,save_path,save_name,model_name="Zoobot - Hubble"):
    data = pd.read_csv(save_path+save_name)

    questions = list(schema.question_answer_pairs.keys())
    answers = schema.label_cols
    answer_indices = schema.question_index_groups

    print("Calculating deviations...")

    #Get number of volunteers
    number_votes = data[trafo_total[questions[0]]]
    half_num_votes = np.array(number_votes)/2

    #Calculate deviations
    deviations_answers, num_answers = [], []
    labels_complete, predictions_complete = [], []
    for i in range(len(answers)):
        predictions, labels = [], []
        #print("Answer:",answers[i])
        question = answers[i].split("_")[0]
        for j in range(len(data)):
            if data[trafo_total[question]].iloc[j] > half_num_votes[j]:
                predictions.append(data[answers[i]+"_pred_frac"].iloc[j])
                labels.append(data[trafo_answers[answers[i]]].iloc[j])
        #print(" --> Predictions:",predictions[:5])
        #print(" --> Labels:",labels[:5])
        deviations = np.abs(np.array(predictions)-np.array(labels))
        #print(" --> Deviations:",deviations[:5])
        num_answers.append(len(deviations))
        deviations_answers.append(np.mean(deviations))
        labels_complete.append(labels)
        predictions_complete.append(predictions)
    
    #Plot
    plt.figure(figsize=(8,12))
    for i in range(len(answer_indices)):
        plt.barh(answers[answer_indices[i][0]:answer_indices[i][1]+1],deviations_answers[answer_indices[i][0]:answer_indices[i][1]+1])
        plt.text(deviations_answers[answer_indices[i][0]:answer_indices[i][1]+1][0] + 0.003,\
                    answer_indices[i][0] + .25,str(num_answers[answer_indices[i][0]:answer_indices[i][1]+1][0]),fontsize="small",color="gray")
    plt.gca().invert_yaxis()
    plt.margins(0.01)
    plt.xlabel("Vote Fraction Mean Deviation")
    plt.grid(axis="x")
    plt.title(model_name)
    plt.savefig(save_path+"deviations_half.png",bbox_inches='tight')
    plt.show()

    return predictions_complete,labels_complete,deviations_answers,num_answers

def apply_model_regression(dataset,run_number,schema,trafo_answers,trafo_total):
    """
        Finetunes the pretrained Zoobot model to a given binary problem and evaluates the performance.
        Predicts number of volunteers choosing certain answer to a binary question for all binary questions simultaneously.
    """
    print("--------- ZOOBOT - Regression --------")
    print("____________________________________\n")
    save_path = "/content/drive/MyDrive/MPE/2022_Ben_Aussel/Results/regression_hubble/full_decision_tree/run_{}/".format(run_number)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("APPLY MODEL TO QUESTIONS: {} ".format(list(schema.question_answer_pairs.keys())))
    print("Saving all results to:",save_path)
    print("____________________________________\n")
    print("---- 1) CREATE DATASETS ----")
    print("____________________________________\n")
    train_dataset, val_dataset, paths_train, paths_val = prepare_dataset(dataset,schema,trafo_answers,trafo_total)
    print("____________________________________\n")
    print("---- 2) PREPARE MODEL ----")
    print("____________________________________\n")
    model = prepare_model_votes(schema)
    print("____________________________________\n")
    print("---- 3) TRAIN MODEL ----")
    print("____________________________________\n")
    trained_model = train_finetuning_loop(model,schema,train_dataset,val_dataset,save_path,loop_yes=False)
    print("____________________________________\n")
    print("---- 4) MAKE PREDICTIONS ON DATASETS ----")
    print("____________________________________\n")
    predict_on_dataset_reg(paths_val,trained_model,save_path,dataset,schema,save_name="val_predictions.csv")
    mean_deviation_half(schema,trafo_answers,trafo_total,save_path,save_name="val_predictions.csv")
    predict_on_dataset_reg(list(dataset["id_str"]),trained_model,save_path,dataset,schema)

hubble_schema = schemas.Schema(label_metadata.gz_hubble_pairs,label_metadata.gz_hubble_dependencies)

apply_model_regression(df,"1",hubble_schema,gz_hubble.gz_hubble_trafo_answers,gz_hubble.gz_hubble_trafo_total)