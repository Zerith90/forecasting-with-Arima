# Databricks notebook source

import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
import itertools
import numpy as np

# COMMAND ----------

# MAGIC %sql
# MAGIC SET
# MAGIC hive.cli.print.header = FALSE
# MAGIC ;
# MAGIC 
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_impressions
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         tmp_query_generator_dell_site_wide_visit_attribution_impressions AS SELECT
# MAGIC *
# MAGIC             FROM
# MAGIC                
# MAGIC                             mm_impressions imps
# MAGIC                         WHERE
# MAGIC                             organization_id IN (100991)
# MAGIC                             AND mm_uuid != '00000000-0000-0000-0000-000000000000'
# MAGIC                             AND impression_date BETWEEN date_sub('2017-06-01',1) AND date_add('2018-03-01',1);
# MAGIC 
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_clicks
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         tmp_query_generator_dell_site_wide_visit_attribution_clicks AS SELECT
# MAGIC                 DISTINCT mm_uuid
# MAGIC                 ,imp_auction_id
# MAGIC             FROM
# MAGIC                 mm_attributed_events
# MAGIC             WHERE
# MAGIC                 organization_id IN (100991)
# MAGIC                
# MAGIC                 AND mm_uuid != '00000000-0000-0000-0000-000000000000'
# MAGIC                 AND event_date BETWEEN date_sub('2017-06-01',1) AND date_add('2018-03-01',1)
# MAGIC                 AND event_type = 'click'
# MAGIC ;          
# MAGIC 
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_imp_full
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         tmp_query_generator_dell_site_wide_visit_attribution_imp_full AS SELECT
# MAGIC                 -- sometimes the click and impression have different mm_uuids
# MAGIC                 CASE
# MAGIC                     WHEN click.imp_auction_id IS NULL THEN imp.mm_uuid
# MAGIC                     ELSE click.mm_uuid
# MAGIC                 END AS mm_uuid
# MAGIC                 ,imp.auction_id
# MAGIC                 ,imp.timestamp_gmt
# MAGIC                 ,imp.report_timestamp
# MAGIC                 ,imp.impression_date
# MAGIC                 ,imp.organization_id AS organization_id
# MAGIC                 ,imp.organization_name AS organization_name
# MAGIC                 ,imp.agency_id AS agency_id
# MAGIC                 ,imp.agency_name AS agency_name
# MAGIC                 ,imp.advertiser_id AS advertiser_id
# MAGIC                 ,imp.advertiser_name AS advertiser_name
# MAGIC                 ,imp.campaign_id AS campaign_id
# MAGIC                 ,imp.campaign_name AS campaign_name
# MAGIC                 ,imp.strategy_id AS strategy_id
# MAGIC                 ,imp.strategy_name AS strategy_name
# MAGIC                 ,imp.concept_id AS concept_id
# MAGIC                 ,imp.concept_name AS concept_name
# MAGIC                 ,imp.creative_id AS creative_id
# MAGIC                 ,imp.creative_name AS creative_name
# MAGIC                 ,imp.exchange_id AS exchange_id
# MAGIC                 ,imp.exchange_name AS exchange_name
# MAGIC                 ,imp.width AS width
# MAGIC                 ,imp.height AS height
# MAGIC                 ,imp.site_url AS site_url
# MAGIC                 ,imp.day_of_week AS day_of_week
# MAGIC                -- ,imp.week_hour_part AS week_hour_part
# MAGIC                 ,imp.mm_creative_size AS mm_creative_size
# MAGIC                 ,imp.placement_id AS placement_id
# MAGIC                 ,imp.deal_id AS deal_id
# MAGIC                 ,imp.country_id AS country_id
# MAGIC                 ,imp.country AS country
# MAGIC                 ,imp.region_id AS region_id
# MAGIC                 ,imp.region AS region
# MAGIC                 ,imp.dma_id AS dma_id
# MAGIC                 ,imp.dma AS dma
# MAGIC                 ,imp.zip_code_id AS zip_code_id
# MAGIC                 ,imp.zip_code AS zip_code
# MAGIC                 ,imp.conn_speed_id AS conn_speed_id
# MAGIC                 ,imp.conn_speed AS conn_speed
# MAGIC                 ,imp.isp_id AS isp_id
# MAGIC                 ,imp.isp AS isp
# MAGIC                 ,imp.publisher_id AS publisher_id
# MAGIC                 ,imp.site_id AS site_id
# MAGIC                 ,imp.watermark AS watermark
# MAGIC                 ,imp.fold_position AS fold_position
# MAGIC                 ,imp.user_frequency AS user_frequency
# MAGIC                 ,imp.browser_id AS browser_id
# MAGIC                 ,imp.browser AS browser
# MAGIC                 ,imp.os_id AS os_id
# MAGIC                 ,imp.os AS os
# MAGIC                 ,imp.browser_language_id AS browser_language_id
# MAGIC                 ,imp.week_part AS week_part
# MAGIC                 ,imp.day_part AS day_part
# MAGIC                 ,imp.day_hour AS day_hour
# MAGIC                 ,imp.week_part_hour AS week_part_hour
# MAGIC                 ,imp.hour_part AS hour_part
# MAGIC                 ,imp.week_part_hour_part AS week_part_hour_part
# MAGIC                 ,imp.week_hour AS week_hour
# MAGIC                 ,imp.batch_id AS batch_id
# MAGIC                 ,imp.device
# MAGIC                 ,CASE
# MAGIC                     WHEN click.imp_auction_id IS NULL THEN 'V'
# MAGIC                     ELSE 'C'
# MAGIC                 END AS pv_pc_flag
# MAGIC             FROM
# MAGIC                 tmp_query_generator_dell_site_wide_visit_attribution_impressions imp LEFT OUTER JOIN tmp_query_generator_dell_site_wide_visit_attribution_clicks click
# MAGIC                     ON click.imp_auction_id = imp.auction_id
# MAGIC ;
# MAGIC 
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_events
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         tmp_query_generator_dell_site_wide_visit_attribution_events AS SELECT
# MAGIC *,
# MAGIC                 lead (
# MAGIC                     timestamp_gmt
# MAGIC                     ,1
# MAGIC                     ,timestamp_gmt
# MAGIC                 ) over (
# MAGIC                     partition BY mm_uuid
# MAGIC                     ,pixel_id
# MAGIC                 ORDER BY
# MAGIC                     timestamp_gmt DESC
# MAGIC                 ) AS previous_pixel_fire
# MAGIC             FROM
# MAGIC                 mm_events
# MAGIC             WHERE
# MAGIC                 organization_id IN (100991)
# MAGIC                AND pixel_id IN (1021447)
# MAGIC             --  and event_type='conversion'
# MAGIC                 AND mm_uuid != '00000000-0000-0000-0000-000000000000'
# MAGIC                 AND event_date BETWEEN date_sub('2017-06-01',1) AND date_add('2018-03-01',7);
# MAGIC                
# MAGIC             

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_join
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         tmp_query_generator_dell_site_wide_visit_attribution_join AS SELECT
# MAGIC                 imp.timestamp_gmt AS impression_timestamp_gmt
# MAGIC                 ,ev.timestamp_gmt AS event_timestamp_gmt
# MAGIC                 ,to_date (ev.timestamp_gmt) AS event_date
# MAGIC                 ,from_unixtime( unix_timestamp( ev.timestamp_gmt ) + unix_timestamp( imp.report_timestamp ) - unix_timestamp( imp.timestamp_gmt ) ) AS event_report_timestamp
# MAGIC                 ,imp.auction_id AS imp_auction_id
# MAGIC                 ,imp.mm_uuid AS mm_uuid
# MAGIC                 ,imp.organization_id AS organization_id
# MAGIC                 ,imp.organization_name AS organization_name
# MAGIC                 ,imp.agency_id AS agency_id
# MAGIC                 ,imp.agency_name AS agency_name
# MAGIC                 ,imp.advertiser_id AS advertiser_id
# MAGIC                 ,imp.advertiser_name AS advertiser_name
# MAGIC                 ,imp.campaign_id AS campaign_id
# MAGIC                 ,imp.campaign_name AS campaign_name
# MAGIC                 ,imp.strategy_id AS strategy_id
# MAGIC                 ,imp.strategy_name AS strategy_name
# MAGIC                 ,imp.concept_id AS concept_id
# MAGIC                 ,imp.concept_name AS concept_name
# MAGIC                 ,imp.creative_id AS creative_id
# MAGIC                 ,imp.creative_name AS creative_name
# MAGIC                 ,imp.exchange_id AS exchange_id
# MAGIC                 ,imp.exchange_name AS exchange_name
# MAGIC                 ,imp.device
# MAGIC                 ,ev.pixel_id AS pixel_id
# MAGIC                 ,ev.pixel_name AS pixel_name
# MAGIC                 ,imp.pv_pc_flag AS pv_pc_flag
# MAGIC                 ,CASE
# MAGIC                     imp.pv_pc_flag
# MAGIC                     WHEN 'V' THEN unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( imp.timestamp_gmt )
# MAGIC                     WHEN 'C' THEN NULL
# MAGIC                 END AS pv_time_lag
# MAGIC                 ,CASE
# MAGIC                     imp.pv_pc_flag
# MAGIC                     WHEN 'V' THEN NULL
# MAGIC                     WHEN 'C' THEN unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( imp.timestamp_gmt )
# MAGIC                 END AS pc_time_lag
# MAGIC                 ,imp.width AS width
# MAGIC                 ,imp.height AS height
# MAGIC                 ,imp.site_url AS site_url
# MAGIC                 ,ev.v1 as mm_v1
# MAGIC                 ,ev.v2 as mm_v2
# MAGIC                 ,ev.v3 as mm_v3
# MAGIC                 ,ev.s1 as mm_s1
# MAGIC                 ,ev.s2 as mm_s2
# MAGIC                 ,ev.s3 as mm_s3
# MAGIC                 ,imp.day_of_week AS day_of_week
# MAGIC                -- ,imp.week_hour_part AS week_hour_part
# MAGIC                 ,imp.mm_creative_size AS mm_creative_size
# MAGIC                 ,imp.placement_id AS placement_id
# MAGIC                 ,imp.deal_id AS deal_id
# MAGIC                 ,imp.country_id AS country_id
# MAGIC                 ,imp.country AS country
# MAGIC                 ,imp.region_id AS region_id
# MAGIC                 ,imp.region AS region
# MAGIC                 ,imp.dma_id AS dma_id
# MAGIC                 ,imp.dma AS dma
# MAGIC                 ,imp.zip_code_id AS zip_code_id
# MAGIC                 ,imp.zip_code AS zip_code
# MAGIC                 ,imp.conn_speed_id AS conn_speed_id
# MAGIC                 ,imp.conn_speed AS conn_speed
# MAGIC                 ,imp.isp_id AS isp_id
# MAGIC                 ,imp.isp AS isp
# MAGIC                 ,imp.publisher_id AS publisher_id
# MAGIC                 ,imp.site_id AS site_id
# MAGIC                 ,imp.watermark AS watermark
# MAGIC                 ,imp.fold_position AS fold_position
# MAGIC                 ,imp.user_frequency AS user_frequency
# MAGIC                 ,imp.browser_id AS browser_id
# MAGIC                 ,imp.browser AS browser
# MAGIC                 ,imp.os_id AS os_id
# MAGIC                 ,imp.os AS os
# MAGIC                 ,imp.browser_language_id AS browser_language_id
# MAGIC                 ,imp.week_part AS week_part
# MAGIC                 ,imp.day_part AS day_part
# MAGIC                 ,imp.day_hour AS day_hour
# MAGIC                 ,imp.week_part_hour AS week_part_hour
# MAGIC                 ,imp.hour_part AS hour_part
# MAGIC                 ,imp.week_part_hour_part AS week_part_hour_part
# MAGIC                 ,imp.week_hour AS week_hour
# MAGIC                 ,imp.batch_id AS batch_id
# MAGIC                 ,row_number (
# MAGIC                 ) over (
# MAGIC                     partition by ev.pixel_id,imp.mm_uuid, ev.timestamp_gmt order by imp.pv_pc_flag, imp.timestamp_gmt desc 
# MAGIC               
# MAGIC                 ) AS rank
# MAGIC             FROM
# MAGIC                 tmp_query_generator_dell_site_wide_visit_attribution_imp_full imp JOIN tmp_query_generator_dell_site_wide_visit_attribution_events ev
# MAGIC                 join t1_meta_campaign c
# MAGIC                     on ev.mm_uuid = imp.mm_uuid
# MAGIC             WHERE
# MAGIC                 (
# MAGIC                     (
# MAGIC                         imp.pv_pc_flag = 'C'
# MAGIC                         AND unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( imp.timestamp_gmt ) BETWEEN 0 AND c.pc_window_minutes
# MAGIC                     )
# MAGIC                     OR (
# MAGIC                         imp.pv_pc_flag = 'V'
# MAGIC                         AND unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( imp.timestamp_gmt ) BETWEEN 0 AND c.pv_window_minutes
# MAGIC                     )
# MAGIC                 )
# MAGIC                 AND (
# MAGIC                     unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( ev.previous_pixel_fire ) >= 60
# MAGIC                     OR unix_timestamp( ev.timestamp_gmt ) - unix_timestamp( ev.previous_pixel_fire ) = 0
# MAGIC                 )
# MAGIC 
# MAGIC ;
# MAGIC 
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS dell_tmp_query_generator_dell_site_wide_visit_attribution_conversion
# MAGIC ;
# MAGIC 
# MAGIC CREATE
# MAGIC     TABLE
# MAGIC         dell_tmp_query_generator_attribution_conversion AS SELECT
# MAGIC                 *
# MAGIC             FROM
# MAGIC                 tmp_query_generator_dell_site_wide_visit_attribution_join
# MAGIC             WHERE
# MAGIC                 rank = 1
# MAGIC ;
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_impressions;
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_clicks;
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_imp_full;
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_events   ;           
# MAGIC DROP
# MAGIC     TABLE
# MAGIC         IF EXISTS tmp_query_generator_dell_site_wide_visit_attribution_join;

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC  a.*,conversion from
# MAGIC    (select
# MAGIC         to_date(report_timestamp) as mm_date,
# MAGIC         campaign_name,
# MAGIC       campaign_id,
# MAGIC         sum(total_spend_cpm/1000) as total_spend
# MAGIC       from
# MAGIC         mm_impressions
# MAGIC       where 
# MAGIC         organization_id =100991
# MAGIC       and
# MAGIC         impression_date between '2017-06-01' and '2018-03-01'
# MAGIC       group by 
# MAGIC         to_date(report_timestamp),
# MAGIC     campaign_name,
# MAGIC       campaign_id
# MAGIC       ) a
# MAGIC     join
# MAGIC     (
# MAGIC       select
# MAGIC         to_date(event_report_timestamp) as mm_date,
# MAGIC       campaign_name,
# MAGIC       campaign_id,
# MAGIC         count(*) as conversion
# MAGIC         from
# MAGIC         mm_attributed_events
# MAGIC      
# MAGIC       where pv_pc_flag='C'  and pixel_id =1021447
# MAGIC         group by to_date(event_report_timestamp),
# MAGIC     campaign_name,
# MAGIC       campaign_id) b
# MAGIC         on a.mm_date=b.mm_date and a.campaign_id=b.campaign_id

# COMMAND ----------

results_df = sqlContext.sql('''

 select
 a.*,conversion from
   (select
        to_date(report_timestamp) as mm_date,
        sum(total_spend_cpm/1000) as total_spend
      from
        mm_impressions
      where 
        organization_id =100991
      and
        impression_date between '2017-06-01' and '2018-03-01'
      group by 
        to_date(report_timestamp)
      ) a
    join
    (
      select
        to_date(event_report_timestamp) as mm_date,
        count(*) as conversion
        from
        mm_attributed_events
        where pv_pc_flag='C'
        and pixel_id =1021447
        group by to_date(event_report_timestamp)
       ) b
        on a.mm_date=b.mm_date
''').toPandas()




# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Inspect the data

# COMMAND ----------

#results_df.head()
results_df.tail(25)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set a 2/3 training and 1/3 test data

# COMMAND ----------

#sort by date
results_sort_by_date = results_df.sort_values(by="mm_date")

#add a new column to denote the cost per visit
results_sort_by_date["cost_per_site_visit"] = results_sort_by_date["total_spend"]/ results_sort_by_date["conversion"]

#show results
results_sort_by_date.tail(25)

# COMMAND ----------

pd.set_option('display.notebook_repr_html', True)
size = int(len(results_sort_by_date) * 0.66)
print(size)
#rename conversions to site_visits
results_sort_by_date =results_sort_by_date.rename(columns={'conversion':'site_visits'})
#get number of site visits 1st

site_visits = results_sort_by_date.filter(['mm_date','site_visits'])
#set the training and test data
# train,test = site_visits[0:size],site_visits[size:len(site_visits)]
# train
# site_visits = sqlContext.createDataFrame(site_visits)
# site_visits.registerTempTable('site_visits')
# display(sqlContext.sql("select * from site_visits"))

# COMMAND ----------



a=sqlContext.createDataFrame(results_sort_by_date).show()
display(a)

# COMMAND ----------

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# COMMAND ----------

def evaluate_models(dataset, p_values, d_values, q_values):
#   print(dataset)
  dataset = dataset.astype('float32')
  best_score, best_cfg = float("inf"), None
  for p in p_values:
      for d in d_values:
          for q in q_values:
              order = (p,d,q)
              try:
                  mse = evaluate_arima_model(dataset, order)
                  if mse < best_score:
                      best_score, best_cfg = mse, order
                  print('ARIMA%s MSE=%.3f' % (order,mse))
              except:
                  continue
  print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# COMMAND ----------

# print(site_visits.values[:,1])
# evaluate parameters
p_values = [1]
d_values = range(0, 1)
q_values = range(0,1)
evaluate_models(site_visits.values[:,1], p_values, d_values, q_values)


# COMMAND ----------



# COMMAND ----------

X=site_visits.values[:,1]
X = X.astype('float32')
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:]
history = [x for x in train]
# make predictions
predictions = list()
for t in range(len(test)):
  model = ARIMA(history, order=(1,0,1))
  model_fit = model.fit(disp=0)
  yhat = model_fit.forecast()[0]
  predictions.append(yhat)
  history.append(test[t])
  print('>predicted=%.3f, expected=%.3f' % (yhat, test[t]))
# calculate out of sample error
error = mean_squared_error(test, predictions)
print(error)

# COMMAND ----------

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
forecast = model_fit.forecast()[0]
# forecast = model_fit.predict(start='2018-03-01', end='2018-03-02')
forecast = model_fit.forecast(steps=30)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, 180)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1

# COMMAND ----------

print(model_fit.summary())

# COMMAND ----------

site_visits.values[:,1]
# model = ARIMA(site_visits.values[:,1], order=(5,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# COMMAND ----------

print("lol")

# COMMAND ----------

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# COMMAND ----------

