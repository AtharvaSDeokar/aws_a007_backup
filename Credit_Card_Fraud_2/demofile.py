
from kafka import KafkaConsumer

consumer = KafkaConsumer('my_topic', bootstrap_servers='localhost:9092')

for message in consumer:
    print(message)

consumer.close()
