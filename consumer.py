import pika
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Configure RabbitMQ connection
rabbitmq_host = 'localhost'
connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
channel = connection.channel()
channel.queue_declare(queue='question_queue')

# Define the callback function to consume messages
def callback(ch, method, properties, body):
    logger.info(f"Received message: {body.decode('utf-8')}")

# Set up consumer
channel.basic_consume(queue='question_queue', on_message_callback=callback, auto_ack=True)

logger.info('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
