# streaming word count using spark dstreams

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

conf = SparkConf()\
    .setMaster('local[2]')\
    .setAppName('StreamingWordCount')

sc = SparkContext(conf=conf)

# stream processing with micro-batch duration = 10 seconds
ssc = StreamingContext(sc, batchDuration=10)

# define the source = socket client (that receive data from server)
stream = ssc.socketTextStream("13.234.75.86", 4444)

# stream operations (transformations)
result = stream.map(lambda line: line.lower())\
    .flatMap(lambda line: line.split())\
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


### steps for execution
# step 1: run a server socket on port 4444
# terminal> ncat -k -l 4444
# step 2: start this application.
# step 3: start writing contents on listening/server socket.

# the application will start processing data
# and produce output continuously after every 10 seconds


