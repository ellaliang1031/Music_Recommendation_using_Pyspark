#question1
#(a)
#draw the directory
hdfs dfs -ls -R hdfs:///data/shared/msd | awk '{print $8}' | sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'
#(b)
hdfs dfs -du -h  hdfs:///data/shared/msd/audio/attributes
#(c)
for i in `hdfs dfs -ls -R hdfs:///data/shared/msd/ | awk '{print $8}'`; do echo $i ; hdfs dfs -cat $i | wc -l; done > ~/q1ccount.txt
hadoop com.sun.tools.javac.Main LineCount.java
jar cf lc.jar LineCount*.class
hadoop jar lc.jar LineCount hdfs:///data/shared/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/ q1c1.txt
hadoop fs -cat q1c1.txt/part-r-00000

#question2
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *
#(a)
#read tasteprofile
tasteprofile_sch = StructType([
    StructField("USER", StringType(), True),
    StructField("SONG", StringType(), True),
    StructField("PLAYCOUNTS", IntegerType(), True),
])

tasteprofile = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", False)
    .option("delimiter", "\t")
    .option("inferSchema", False)
    .schema(tasteprofile_sch)
    .load("hdfs:///data/shared/msd/tasteprofile/triplets.tsv")
) 

#read mismatches
mismatches1 = spark.read.text('hdfs:///data/shared/msd/tasteprofile/mismatches/sid_mismatches.txt/')
mismatchfile1 = mismatches1.select((F.substring(F.col('VALUE'), 9, 18)).alias('MISSONG'), (F.substring(F.col('VALUE'), 28, 18)).alias('TRACKS'))
mismatches2 = spark.read.text('hdfs:///data/shared/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt/')
mismatchfile2 = mismatches2.select((F.substring(F.col('VALUE'), 11, 18)).alias('MISSONG'), (F.substring(F.col('VALUE'), 30, 18)).alias('TRACKS'))
mismatchfile = mismatchfile1.union(mismatchfile2)
#find the matched song
tasteprofile1 = tasteprofile.join(mismatchfile, tasteprofile.SONG == mismatchfile.MISSONG, 'left').filter(F.col('TRACKS').isNull()).drop('TRACKS')
tasteprofile1.count() ##45785819
tasteprofile2 = tasteprofile.join(mismatchfile1, tasteprofile.SONG == mismatchfile1.MISSONG, 'left').filter(F.col('TRACKS').isNull()).drop('TRACKS')
tasteprofile2.count()##45795100
#(b)
#load the attributes
#msd-rh-v1.0.attributes.csv
attributesrh_sch = StructType([
    StructField("FEATURES", StringType(), False),
    StructField("TYPES", StringType(), False),
])

attributes_rh = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", False)
    .option("inferSchema", False)
    .schema(attributesrh_sch)
    .load("hdfs:///data/shared/msd/audio/attributes/msd-rh-v1.0.attributes.csv")
)
attributes_rh.select('TYPES').distinct().show()   #string and numeric
attributes_rh = attributes_rh.replace(['numeric', 'string'],['DoubleType()','StringType()'])

typesrh = {
    'STRING': 'StringType()',
    'NUMERIC': 'DoubleType()'
}
featurerh_sch = StructType([
    StructField(FEATURES, eval(typesrh[TYPES]), True) for (FEATURES, TYPES) in attributes_rh.rdd.collect()
])

# Read data
featurerh = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", False)
    .option("inferSchema", False)
    .schema(featurerh_sch)
    .load("hdfs:///data/shared/msd/audio/features/msd-rh-v1.0.csv")
)
featurerh.select('component_1', 'instancename').show(10)
featurerh.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/featurerh.csv')
hdfs dfs -getmerge hdfs:///home/cli138/data/outputs/ghcnd/featurerh.csv ~/featurerh.csv

#Audio Similarity
#QUESTION1
# select numerical variables
str_cols = [i.name for i in featurerh.schema.fields if "StringType" in str(i.dataType)]
str_label = featurerh.select(*str_cols)
num_feature = featurerh.drop(*str_cols)
num_feature.select(num_feature.columns[:6]).describe().show()

from pyspark.mllib.stat import Statistics 
import pandas as pd 

def compute_correlation_matrix(df, method='pearson'):
	df_rdd = df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(df_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=df.columns, 
                    index=df.columns)
    return corr_mat_df



cor_matrix = compute_correlation_matrix(num_feature.dropna())
cor_matrix.head()

def high_correlation(cor_matrix):
	high_correlation = []
	newframe = pd.DataFrame(cor_matrix)
	for index, row in newframe.iterrows():   
        for col_name in newframe.columns:
            if 1 > row[col_name] > 0.7:
            	 high_correlation.append((index, col_name))
    return high_correlation

high_correlation = high_correlation(cor_matrix)
high_correlation#36



#(b)load MAGD and select songs matched
magd_sch = StructType([
    StructField("MASGTRACKS", StringType(), True),
    StructField("GENRELABEL", StringType(), True),
])

genreassign = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("inferSchema", False)
    .schema(magd_sch)
    .load("hdfs:///data/shared/msd/genre/msd-MAGD-genreAssignment.tsv")
)

genreassign1 = (genreassign.select('MASGTRACKS').subtract(mismatchfile1.select('TRACKS'))).withColumnRenamed('TRACKS', 'MATCH')
genreassign1.count()#415350
genreassign = genreassign.join(mismatchfile1, genreassign.MASGTRACKS == mismatchfile1.TRACKS, 'left').filter(F.col('TRACKS').isNull())
genreassign.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/magdfinal1.csv')
hdfs dfs -getmerge hdfs:///home/cli138/data/outputs/ghcnd/magdfinal1.csv ~/magdfinal1.csv

#(c)merge
labelassign = featurerh.join(genreassign, genreassign.MASGTRACKS == featurerh.instanceName.substr(2,18), 'left').filter(F.col('GENRELABEL').isNotNull())
labelassign.show(5)
labelassignshow = labelassign.select('instanceName', 'GENRELABEL')
labelassignshow.show(5)
labelassign = labelassign.drop('MISSONG', 'TRACKS', 'instanceName')
#question2
#(a)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
tofeature = VectorAssembler(inputCols = num_feature.columns, outputCol = "features")
df = tofeature.transform(labelassign)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(df) 
scaled = scaler.transform(df)
sample = scaled.select('MASGTRACKS', F.col('scaledFeatures').alias('features'), 'GENRELABEL')
# PCA
from pyspark.ml.feature import PCA
pca = PCA(k=60, inputCol="features", outputCol="pca_Features")
model = pca.fit(sample)
model.explainedVariance
pca = PCA(k=30, inputCol="features", outputCol="pca_Features_3")
model = pca.fit(sample)
pcamodel = model.transform(sample)

#(b)
def elec_or_not(songtype):
    if songtype == 'Electronic':
    	return 1
    else:
    	return 0
elec_or_not_udf = F.udf(lambda songtype: elec_or_not(songtype), IntegerType())
elecsample =sample.withColumn('elec_or_not', elec_or_not_udf(F.col('GENRELABEL')))
num_elec = elecsample.filter(F.col('elec_or_not') == 1)
#num_elec.count() #40048
num_not_elec = elecsample.filter(F.col('elec_or_not') == 0)
#num_not_elec.count() #372982
##the whole dataset for random forest
elecsampleall =df.withColumn('elec_or_not', elec_or_not_udf(F.col('GENRELABEL')))
#pca dataset
elecsamplepca = pcamodel.withColumn('elec_or_not', elec_or_not_udf(F.col('GENRELABEL')))
#(c)
from pyspark.sql.functions import count, udf
from pyspark.sql.types import BooleanType
from operator import truediv

counts = (elecsample
    .groupBy(col("elec_or_not"))
    .agg(count("*").alias("n"))
    .rdd.map(lambda r: (r.elec_or_not, r.n))
    .collectAsMap()) 

fractions = elecsample.select("elec_or_not").distinct().withColumn("fraction", F.lit(0.7)).rdd.collectAsMap()
sampled = elecsample.stat.sampleBy("elec_or_not", fractions=fractions, seed=1)
testsample = elecsample.subtract(sampled)
#for the whole dataset
fractionsall = elecsampleall.select("elec_or_not").distinct().withColumn("fraction", F.lit(0.7)).rdd.collectAsMap()
sampledall = elecsampleall.stat.sampleBy("elec_or_not", fractions=fractionsall, seed=1)
testsampleall = elecsampleall.subtract(sampledall)
#
fractionspca = elecsamplepca.select("elec_or_not").distinct().withColumn("fraction", F.lit(0.7)).rdd.collectAsMap()
sampledpca = elecsamplepca.stat.sampleBy("elec_or_not", fractions=fractionspca, seed=1)
testsamplepca = elecsamplepca.subtract(sampledpca)
#(d)
# Logistic Regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
lr = LogisticRegression(maxIter=10, regParam=0.001, elasticNetParam=0.5, threshold = 0.3, family="binomial", featuresCol="features", labelCol="elec_or_not")
lrmodel = lr.fit(sampled)
lrpred = lrmodel.transform(testsample)
eva = lrpred.groupBy('prediction').agg(F.count('MASGTRACKS'))
eva.show()
lrpred.filter((F.col('prediction') == 1) & (F.col('elec_or_not') == 1)).count()

metriclist = ['weightedPrecision', 'weightedRecall', 'accuracy']
for metric in metriclist:
    assess = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="elec_or_not", metricName=metric)
    result = assess.evaluate(lrpred)
    print('{}: {}'.format(metric, result))

#RandomForest
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=20, featuresCol="features", labelCol="elec_or_not", seed=1)
rfmodel = rf.fit(sampledall)
rfpred = rfmodel.transform(testsampleall)
eva2 = rfpred.groupBy('prediction').agg(F.count('MASGTRACKS'))
eva2.show()

metriclist = ['weightedPrecision', 'weightedRecall', 'accuracy']
for metric in metriclist:
    assess = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="elec_or_not", metricName=metric)
    result = assess.evaluate(rfpred)
    print('{}: {}'.format(metric, result))

#SVM
from pyspark.ml.classification import LinearSVC
svm = LinearSVC(maxIter=10, regParam=0.01, threshold = 0.3, featuresCol="features", labelCol="elec_or_not")
svmmodel = svm.fit(sampledpca)
svmpred = svmmodel.transform(testsamplepca)
eva3 = svmpred.groupBy('prediction').agg(F.count('MASGTRACKS'))
eva3.show()
metriclist = ['weightedPrecision', 'weightedRecall', 'accuracy']
for metric in metriclist:
    assess = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="elec_or_not", metricName=metric)
    result = assess.evaluate(svmpred)
    print('{}: {}'.format(metric, result))

#question3
#(a)
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StringIndexer
featuremul = StringIndexer(inputCol='GENRELABEL', outputCol='Index')
transformer = featuremul.fit(sample)
scaledata = transformer.transform(sample)
scaledata = scaledata.drop('elec_or_not')


#(b)

fraction_s = scaledata.select("Index").distinct().withColumn("fraction", F.lit(0.7)).rdd.collectAsMap()
sampled_s = scaledata.stat.sampleBy("Index", fractions=fraction_s, seed=1)
testsample_s = scaledata.subtract(sampled_s)

##
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
svmmodelmulti = LinearSVC(maxIter=5, regParam=0.1)
ovr = OneVsRest(classifier=svmmodelmulti, featuresCol='features', labelCol="Index")
svmmodel_s = ovr.fit(sampled_s)
pred_svm_s = svmmodel_s.transform(testsample_s)

#e)
pred_svm_s.filter(F.col('Index') == F.col('prediction')).count()
multiclass_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="multi_label", metricName='accuracy')
value = multiclass_evaluator.evaluate(pred_svm_s)
print('accuracy ', value)

#Song recommendations
#question1
#(a)
#uniquesong
tasteprofile.select('SONG').distinct().count()#384546
#uniqueuser
tasteprofile.select('USER').distinct().count()#1019318
#(b)
activeuser = tasteprofile.groupBy('USER').agg(F.sum('PLAYCOUNTS').alias('TOTALPLAY')).orderBy('TOTALPLAY', ascending=False)
userforsong = tasteprofile.groupBy('USER').agg(F.countDistinct('SONG').alias('TOTALSONG')).orderBy('TOTALSONG', ascending=False)

#(c)
songdis = tasteprofile.groupBy('SONG').agg(F.sum('PLAYCOUNTS').alias('TOTALPLAY')).orderBy('TOTALPLAY', ascending=False)
activeuser.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/activeuser.csv')
songdis.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/songdis.csv')

hdfs dfs -getmerge hdfs:///home/cli138/data/outputs/ghcnd/activeuser.csv ~/activeuser.csv
hdfs dfs -getmerge hdfs:///home/cli138/data/outputs/ghcnd/songdis.csv ~/songdis.csv
#(d)
userforsong.approxQuantile('TOTALSONG', [0.25], 0)#16
songdis.approxQuantile('TOTALPLAY', [0.25], 0)#8
userac = userforsong.select('USER').filter(F.col('TOTALSONG') > 16)
songpo = songdis.select('SONG').filter(F.col('TOTALPLAY') > 8)
tasteprofile = tasteprofile.join(userac, on = 'USER', how='inner')
tasteprofile = tasteprofile.join(songpo, on = 'SONG', how='inner')#44615167
#(e)
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
uniqtaste = tasteprofile.withColumn('UNIQUSER', (F.count('USER').over(Window.partitionBy('USER')) == 1).cast('integer'))
uniqtaste = uniqtaste.withColumn('UNIQSONG', (F.count('SONG').over(Window.partitionBy('song')) == 1).cast('integer'))

# transform user and song into dummy v
user_d = StringIndexer(inputCol='USER', outputCol='USER_D')
song_d = StringIndexer(inputCol='SONG', outputCol='SONG_D')
uniqtaste = user_d.fit(uniqtaste).transform(uniqtaste)
uniqtaste = song_d.fit(uniqtaste).transform(uniqtaste)
from pyspark.ml.feature import QuantileDiscretizer
qd = QuantileDiscretizer(numBuckets=6, inputCol="PLAYCOUNTS", outputCol="PLAYCOUNTS_T")
uniqtaste = qd.fit(uniqtaste).transform(uniqtaste)
# Split the training and test data sets
trainsam = uniqtaste.filter((F.col('UNIQUSER') == 1) | (F.col('UNIQSONG') == 1)) 
leftover = uniqtaste.filter((F.col('UNIQUSER') == 0) & (F.col('UNIQSONG') == 0))
trainsam2, test = leftover.randomSplit([0.8, 0.2], seed=6)
train = trainsam.union(trainsam2) 
train = train.drop('UNIQSONG', 'UNIQUSER')
test = test.drop('UNIQSONG', 'UNIQUSER')

#QUESTION2
#(a)
from pyspark.ml.recommendation import ALS
als = ALS(rank=10, seed=1, userCol="USER_D", itemCol="SONG_D", ratingCol="PLAYCOUNTS_T")
alsmodel = als.fit(train)
alspred = alsmodel.transform(test)
alspred.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/alsresult1.csv', header=True)
predals = spark.read.csv('hdfs:///home/cli138/data/outputs/ghcnd/alsresult1.csv', header=True)

predals.orderBy('PLAYCOUNTS_T', ascending=False).show(5)
recommend = alsmodel.recommendForAllUsers(3)
recommend.filter(F.col('USER_D') == 3).show()

#c) Metrics
# Precision @ 5
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
window = Window.partitionBy(predals['USER_D']).orderBy(predals['prediction'].desc())
pred_bi = predals.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 5)

# Then we transfer the True and Predicted ratings to 0/1 labels by using the threshold of 3.5
pred = pred_bi.drop('rank')
pred_bi = (pred_bi
			.withColumn('T_LABEL', (F.col('PLAYCOUNTS_T') >= 1.5).cast('integer'))
			.withColumn('P_LABEL', (F.col('prediction') >= 1.5).cast('integer')))

# calculate the precision at 3 :
precision = pred_bi.filter((F.col('P_LABEL') == 1) & (F.col('T_LABEL') == 1)).count() / \
					pred_bi.filter(F.col('P_LABEL') == 1).count() #  0.6525880307149654


#############not run successfully
# NDCG @ 10
recommend_10 = alsmodel.recommendForAllUsers(10)
recommendations = recommend_10.select(F.col("USER_D"), F.explode("recommendations").alias('test'))
window = Window.partitionBy(recommendations['USER_D']).orderBy(recommendations['test'].desc())# Add the ranking in to the recommendation dataframe
recommendations = recommendations.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 10)
recommendations = (recommendations
				.withColumn('SONG_D', recommendations.test['SONG_D'])
				.withColumn('similarity', recommendations.test['rating'])
				.drop('test'))

recommendations = recommendations.join(predals.select('USER_D', 'SONG_D', 'PLAYCOUNTS_T'), 
									on=['USER_D', 'SONG_D'], 
									how='left')

# Set 0 to predicted ratings which do not have a true rating
recommendations = recommendations.withColumn("pred_rating", 
											F.when(recommendations["PLAYCOUNTS_T"].isNull(), 0).
											otherwise(recommendations["similarity"]))
recommendations = recommendations.withColumn("true_rating", 
											F.when(recommendations["PLAYCOUNTS_T"].isNull(), 0).
											otherwise(recommendations["PLAYCOUNTS_T"]))
recommendations = recommendations.select('USER_D', 'SONG_D', 'rank', 'pred_rating', 'true_rating')
recommendations.write.csv('hdfs:///home/cli138/data/outputs/ghcnd/recommendations.csv', header=True)
ndcg_df = recommendations.orderBy('USER_D', 'rank')
recommendations = spark.read.csv('hdfs:///home/cli138/data/outputs/ghcnd/recommendations', header=True)

# Calculate NDCG in pyspark:
def get_dcg(rating, rank):
	sum = 0
	for i in range(len(rating)):
		sum += rating[i]/math.log(rank[i] + 1, 2)
	return sum

def get_idcg(rating):
	sum = 0
	for i in range(len(rating)):
		sum += rating[i]/math.log(2 + i % 10, 2)
	return sum
get_dcg_udf = F.udf(lambda x, y: get_dcg(x, y))
get_idcg_udf = F.udf(lambda x: get_idcg(x))
ndcg_df.select(get_dcg_udf(F.collect_list('pred_rating'), F.collect_list('rank'))) 
ndcg_df.orderBy('USER_D', 'true_rating', ascending=False).select(get_idcg_udf(F.collect_list('true_rating'))).show() 

ndcg_df = ndcg_df.orderBy('USER_D', 'rank')
pred_rating_array = [float(i.pred_rating) for i in ndcg_df.collect()]
ndcg_df = ndcg_df.orderBy('USER_D', 'true_rating', ascending=False)
true_rating_array = [float(i.true_rating) for i in ndcg_df.collect()]
rank_array = [int(i.rank) for i in ndcg_df.collect()]
dcg = get_dcg(pred_rating_array, rank_array) 
idcg = get_dcg(true_rating_array, rank_array) 
ndcg_at_10 = dcg/idcg 
def calculate_ap(accurate):
	p = 0
	for i in range(len(accurate)):
		p += sum(accurate[:i + 1]) / (i + 1)
	return (p / len(accurate))

calculate_ap_udf = F.udf(lambda x: calculate_ap(x))

map_at_5 = binary_pred.select('USER_D', 'SONG_D', 'prediction', 'true_label', 'pred_label')
map_at_5 = map_at_5.withColumn('accurate', (F.col('true_label') == F.col('pred_label')).cast('integer'))
map_at_5 = map_at_5.orderBy('USER_D', 'prediction', ascending=False)
map_at_5 = map_at_5.groupBy('USER_D').agg({'accurate':'collect_list'})
map_at_5 = map_at_5.withColumn('ap', calculate_ap_udf(F.col('collect_list(accurate)')))
