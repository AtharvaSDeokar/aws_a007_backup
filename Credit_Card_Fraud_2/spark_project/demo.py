from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

conf = SparkConf().setMaster("local[2]").setAppName("StreamingWordCount")
sc = SparkContext(conf=conf)

# stream processing with micro-batch duration = 10 seconds
ssc = StreamingContext(sc, 10)

# define the source = socket client (that receive data from Ncat)
stream = ssc.socketTextStream("localhost", 4444)

# stream operations (transformations)
result = stream.flatMap(lambda line: line.split())\
    .map(lambda word: (word, 1))\
    .reduceByKey(lambda a,b: a + b)

# define the sink = console
result.pprint()

# streaming action
ssc.start()

# wait for stream processing to execute
print('waiting for termination...')
ssc.awaitTermination()

# stop streaming and spark context
ssc.stop(True)

