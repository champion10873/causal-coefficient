# Import Python Library
import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run on CPU

import tensorflow.compat.v1 as tf

import ltc_model as ltc
import argparse
tf.compat.v1.disable_eager_execution()

# Import Data
def load_trace():
    
    df = pd.read_csv("data/soiltemperature/agric2A_72.csv")
    wind_direction = df["Wind Direction [deg]"].values.astype(np.float32)
    precipitation = df["Precipitation [mm]"].values.astype(np.float32)
    wind_speed = df["Wind Speed [m/s]"].values.astype(np.float32)
    air_temperature = df["HC Air Temperature [aiC]"].values.astype(np.float32)
    relative_humidity = df["HC Relative Humidity [%]"].values.astype(np.float32)
    dew_point = df["Dew Point [aiC]"].values.astype(np.float32)
    vpd = df["VPD [mbar]"].values.astype(np.float32)
    wind_speed_max = df["Wind Speed Max [m/s]"].values.astype(np.float32)
    soil_moisture_10 = df["EAG Soil Moisture [%] 10cm"].values.astype(np.float32)
    soil_moisture_20 = df["EAG Soil Moisture [%] 20cm"].values.astype(np.float32)
    soil_moisture_30 = df["EAG Soil Moisture [%] 30cm"].values.astype(np.float32)
    soil_moisture_110 = df["EAG Soil Moisture [%] 110"].values.astype(np.float32)
    soil_moisture_120 = df["EAG Soil Moisture [%] 120cm"].values.astype(np.float32)
    soil_temperature_10 = df["Soil Temperature 10cm"].values.astype(np.float32)
    soil_temperature_20 = df["Soil Temperature 20cm"].values.astype(np.float32)
    soil_temperature_30 = df["Soil Temperature 30cm"].values.astype(np.float32)
    
    features_name = ["Wind Direction [deg]", "Precipitation [mm]", "Wind Speed [m/s]", "HC Air Temperature [aiC]", "HC Relative Humidity [%]", "Dew Point [aiC]", "VPD [mbar]", "Wind Speed Max [m/s]", "EAG Soil Moisture [%] 10cm", "EAG Soil Moisture [%] 20cm", "EAG Soil Moisture [%] 30cm", "EAG Soil Moisture [%] 110", "EAG Soil Moisture [%] 120cm", "Soil Temperature 10cm", "Soil Temperature 20cm", "Soil Temperature 30cm"]
    features = np.stack([wind_direction, precipitation, wind_speed, air_temperature, relative_humidity, dew_point, vpd, wind_speed_max, soil_moisture_10, soil_moisture_20, soil_moisture_30, soil_moisture_110, soil_moisture_120, soil_temperature_10, soil_temperature_20, soil_temperature_30])
    
    return features_name, features


# Calculate Causal_Coefficient Using Pearson Algorithm
def causal_coefficient(x, y):
    
    correlation_matrix = np.corrcoef(x, y)
    pearson_coefficient = correlation_matrix[0, 1]

    return pearson_coefficient


# Initial Setting
def process_coefficient(seq_len):
    
    coefficient = []
    coefficient_valid = []
    virtual_y = []
    first_cause = []
    second_cause = []
    
    x, y = load_trace()
    
    yy = np.stack(y, axis=-1)
    
    for i in range(0, len(yy),):
        virtual_y.append(yy[i])
    for i in range(0, seq_len,):
        virtual_y.append(yy[i])
    yy = np.stack(virtual_y, axis=1)
    for s in range(0, y.shape[1],):
        coefficient_vl = 0
        start = s
        end = start + seq_len
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                causal_coe = causal_coefficient(yy[i][start:end], yy[j][start:end])
                if(np.isnan(causal_coe)):
                    causal_coe = 0
                coefficient_vl += np.abs(causal_coe)
        coefficient_valid.append(coefficient_vl)
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            coefficient.append(causal_coefficient(y[i], y[j]))
            first_cause.append(x[i])
            second_cause.append(x[j])
    cause_names = np.stack([first_cause, second_cause], axis=-1)
    
    y = np.stack(y, axis=-1)
    coefficient_valid -= np.mean(coefficient_valid) # normalize
    coefficient_valid /= np.std(coefficient_valid) # normalize

    return y, coefficient_valid, cause_names, coefficient


# Dataset Preparation
def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
        
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class SoilTemperatureData:
    def __init__(self, seq_len=32): # Init Function

        # Browse Dataset
        x, y, names, coes = process_coefficient(seq_len)        
        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=1)

        # Dataset Division
        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]] # Dataset for Validation
        self.valid_y = self.train_y[:, permutation[:valid_size]] # Dataset for Validation
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]] # Dataset for Test
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]] # Dataset for Test
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]] # Dataset for Train
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]] # Dataset for Train
        self.names = names # Variable names
        self.coes = coes # Set of Causal_Coefficient

    # Iterate Function For Training a Machine Learning Model Iteratively.
    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)


class SoilTemperatureModel:
    def __init__(self, model_type, model_size, learning_rate=0.001): # Preparation for Training
        self.model_type = model_type # LTC
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 16])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.model_size = model_size
        head = self.x
        if model_type.startswith("ltc"):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if model_type.endswith("_rk"):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif model_type.endswith("_ex"):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(
                self.wm, head, dtype=tf.float32, time_major=True
            )
            self.constrain_op = self.wm.get_param_constrain_op()
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        target_y = tf.expand_dims(self.target_y, axis=-1) # Predicted Value
        self.y = tf.layers.Dense( # Actual Value
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y)) # Average of Squared Differences. 
        optimizer = tf.train.AdamOptimizer(learning_rate) # Optimizer for Training Model
        self.train_step = optimizer.minimize(self.loss) # Create a Train Step Function

        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        self.sess = tf.InteractiveSession() # Create Interactive TensorFlow Session
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join(
            "results", "soiltemperature", "Cause-effect relationships - agric2A_72.csv"
        )
        
        if not os.path.exists("results/soiltemperature"):
            os.makedirs("results/soiltemperature")
            
        with open(self.result_file, "w") as f:
            f.write(
                "Cause, Effect, Causal_Coefficient\n"
            )

        self.checkpoint_path = os.path.join(
            "tf_sessions", "soiltemperature", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/soiltemperature"):
            os.makedirs("tf_sessions/soiltemperature")

        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, gesture_data, epochs, verbose=True, log_period=50):

        # Initialize
        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        names = gesture_data.names
        coes = gesture_data.coes
        self.save()
        # Iterate Training
        for e in range(epochs):
            # Test & Validate
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.test_x, self.target_y: gesture_data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.valid_x, self.target_y: gesture_data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                        names,
                        coes,
                    )
                    self.save()

            losses = []
            accs = []
            # Train
            for batch_x, batch_y in gesture_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y},
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            # Print Logs
            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))): # Check Issue with Loss Value
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
            names,
            coes,
        ) = best_valid_stats
        # Print to Console Window
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        # Output
        with open(self.result_file, "a") as f:
            for i in range(len(coes)):
                f.write(
                    "{}, {}, {:0.4f}\n".format(
                        names[i][0],
                        names[i][1],
                        coes[i],
                    )
                )


if __name__ == "__main__":
    
    # Create ArgumentParser Object
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ltc")
    parser.add_argument("--log", default=1, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()
    
    soiltemperature_data = SoilTemperatureData() # Browse Data
    model = SoilTemperatureModel(model_type=args.model, model_size=args.size) # Create an Instance of SoilTemperatureModel
    
    model.fit(soiltemperature_data, epochs=args.epochs, log_period=args.log) # Train a Machine Learning Model