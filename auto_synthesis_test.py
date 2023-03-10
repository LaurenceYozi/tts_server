import httpx
import os
import time
from glob import glob

def getEngTTS(text, client):
    server_url = "http://192.168.1.27:8508/"
    data = {"text": text, "gender": "female", "datatype": "wav", "speed": 0.8}
    r = client.post(
        server_url, json=data,
    )

text = [
        "When answering this question, it's important to remember that food is most nutritious at the point of harvest, says Fatima Hachem, Senior Nutrition Officer at the UN's Food and Agriculture Organization. Fresh produce starts degrading as soon as it's picked from the ground or tree, because that ground or tree is its source of nutrients and energy.",
        "Who is that beautiful girl?",
        "Can you talk to Jim? Well, he is busy now.",
        "She works all day in a store; in addition, she takes classes in the evenings.",
        "Stars can't shine without darkness.",
        "The Ukrainian government has been forced to urge the population to try and use energy sparingly as a result.",
        # "What a pity!",
        # "“Right?” she asked.",
        # "Dr. Brown died last night.",
        # "“Where have you been?” she said.",
        # "On December 24, 1999, Melisa Wong left for New York.",
        # "At age 15, students in Taiwan are required to know about 2000 words; by age 18, as many as 7000.",
        # "He asked his mother, “Can I watch TV after dinner?” She answered, “Only if you finish your homework before then.”",
        # "\"Kate, do you need help moving your things?\" \"Yes, thank you!\"",
        # "In 2019 I went to Japan on business.",
        # "so, suddenly, a dog jumped on me and I had to run away!",
        # "Those women are parents of this class.",
        # "The man is working hard on many documents.",
        # "Francis had been tying a bow on his shoes when he realized his shoes were on the wrong feet."
]

client = httpx.Client()
for item in text:
    try:
        getEngTTS(item, client)
        print(f"Synthesis text: {item}")
    except Exception:
        print("Fail")
client.close()

time.sleep(4)
remove_data = glob(os.path.join("wavs/", "*_lowpassed.wav"))
for file in remove_data:
    os.remove(file)
