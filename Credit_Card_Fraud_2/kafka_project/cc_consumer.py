
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer('cc', bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: json.loads(x.decode('utf-8')))

for message in consumer:
    msg = message.value
    print('received: ', msg)
