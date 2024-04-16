import lstm_agent as lstm
import rnn_agent as rnn
import joblib
model = lstm.create_model()
joblib.dump(model, 'model1.pk1')

# rnn.__main__()