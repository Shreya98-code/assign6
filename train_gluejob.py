import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import tweet_prep
from tweet_prep import Preprocessor

my_prep2 = Preprocessor()

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "assign4", table_name = "evaluation", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "assign4", table_name = "evaluation", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")

## @type: Map
## @args: [f = <function>, transformation_ctx = "<transformation_ctx>"]
## @return: <output>
## @inputs: [frame = <frame>]
def map_function(dynamicRecord):
    tweet = dynamicRecord["tweet"]
    features = my_prep2.tweet_process(tweet)
    dynamicRecord["features"] = features
    return dynamicRecord
processed_tweets2 = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "processed_tweets2")


## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://shreya-assign-4/Processed/Evaluation"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = processed_tweets2, connection_type = "s3", connection_options = {"path": "s3://shreya-assign-4/Processed/Evaluation"}, format = "json", transformation_ctx = "datasink2")
job.commit()