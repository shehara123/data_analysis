from flask import Flask,request,jsonify
import pandas as pd
import pickle
#cd "D:\advance analysis  python"
model = pickle.load(open('D:/advance analysis  python/RF.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def predict():
    boarding = request.form.get('boarding')
    Convenience = request.form.get('Convenience')
    service = request.form.get('service')
    comfort = request.form.get('comfort')
    Class_Business = request.form.get('Class_Business')
    Class_Eco = request.form.get('Class_Eco')
    Business_travel = request.form.get('Type.of.Travel_Business travel')
    Personal_Travel = request.form.get('Type.of.Travel_Personal Travel')

    data = [[boarding, Convenience, service, comfort, Class_Business, Class_Eco,
             Business_travel, Personal_Travel]]
    df = pd.DataFrame(data, columns=['boarding', 'Convenience', 'service', 'comfort',
                                     'Class_Business', 'Class_Eco',
                                     'Type.of.Travel_Business travel', 'Type.of.Travel_Personal Travel'])

    result = model.predict(df)[0]
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)



