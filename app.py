from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn
app = Flask(__name__)

@app.route('/')
def index():
	return "Welcome to the RAINFALL PREDICTION API"

@app.route('/predict', methods = ['POST'])
def predict():
	model = pickle.load(open('rainfall_final.pkl' ,'rb'))
	
	try:
		MeanTemp = float(request.form['MeanTemp'])
		maxTemp = float(request.form['MaxTemp'])
		Mintemp = float(request.form['Mintemp'])
		DewPoint = float(request.form['DewPoint'])

		AverageHumidity = float(request.form['AverageHumidity'])
		MaxHumidity = float(request.form['MaxHumidity'])
		MinimumHumidity = float(request.form['MinimumHumidity'])
		SeaLevelPressure = float(request.form['SeaLevelPressure'])

		AverageWindSpeed = float(request.form['AverageWindSpeed'])
		MaximumWindSpeed = float(request.form['MaximumWindSpeed'])
		
		test_vector = np.asanyarray([MeanTemp, maxTemp, Mintemp, DewPoint, AverageHumidity, MaxHumidity, MinimumHumidity, SeaLevelPressure, AverageWindSpeed, MaximumWindSpeed])
		test_vector = np.reshape(test_vector,(1,10))
		# reverse_mapping = ['Barley','Corn-Field for silage','Corn-Field for stover','Millet','Potato','Sugarcane']
		# reverse_mapping = np.asarray(reverse_mapping)
		a = model.predict(test_vector)
		#hell yeah this corresponds to barley
		#prediction = reverse_mapping[a]

		# give to model and predict

		#hell yeah this corresponds to barley
		# prediction = reverse_mapping[a]

		# c=0
		# for ix in range(a.shape[0]):
		#     if a[ix].any() == 1:
		#         ans = c
		#     c+=1

		#print reverse_mapping[ans]
		#print((moisture, nitrogen, phosphorous, potassium))
	

		res = a[0]

		return jsonify({'result' : res })
	except Exception as e:
		print(e)
		return jsonify({'error' : 'Wrong Parameters' })

if __name__ == '__main__':
	app.run(port = 8081, debug = True)
