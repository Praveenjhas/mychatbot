import google.generativeai as genai
genai.configure(api_key="AIzaSyBzltFuAxizZPa6yfgkolS0-5BvlIIOcYI")
for m in genai.list_models():
    print(m.name)