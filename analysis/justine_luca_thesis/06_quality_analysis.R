#Analysis of the quality of the final data
#Analyzing the consensus for both the header data as the climate data
setwd("/Users/justineluca/Documents/thesis LWK/zooniverse/jungleweather-master")

# load the tidyverse for data wrangling
library(tidyverse)
#read in processed data
climate_data <- read.table("data/final_climate_data.csv",
                           header = TRUE,
                           sep = ",",
                           stringsAsFactors = FALSE)

header_data <- read.table("data/final_header_data.csv",
                          header = TRUE,
                          sep = ",",
                          stringsAsFactors = FALSE)


#make table with header data in one row (location, year, month) sorted by file and folder
location <- header_data[header_data$field == "location", ]
year <- header_data[header_data$field == "year", ]
month <- header_data[header_data$field == "month", ]
df_list <- list(location, year, month)

header_new <- Reduce(function(x, y) merge(x, y, by=c("file","folder")), df_list)
header_new <-header_new %>% select(file,folder,value.x,value.y,consensus.y,value,consensus)
names(header_new)[names(header_new) == "value.x"] <- "location"
names(header_new)[names(header_new) == "value.y"] <- "year"
names(header_new)[names(header_new) == "value"] <- "month"
names(header_new)[names(header_new) == "consensus.y"] <- "consensus year"
names(header_new)[names(header_new) == "consensus"] <- "consensus month"

#quality of climate data
#filter "good quality" = consensus >= 0,7
df <- climate_data[climate_data$consensus >= 0.7, ]
#percentage of transcribed values that are of qood quality
percentage_good_quality = length(df$final_value)/length(climate_data$final_value)*100
#68,61% has consensus >= 0,7

# good quality = consensus >= 0,8
df <- climate_data[climate_data$consensus >= 0.8, ]
#percentage of transcribed values that are of qood quality
percentage_good_quality = length(df$final_value)/length(climate_data$final_value)*100
#68,60% has consensus >=0,8
#As there is not much difference between 0,7 and 0,8 as treshold, the highest treshold
#is retained to determine the good quality data.

#merge good quality data with header data
all_data <- merge(x = df, y = header_new, by=c("file","folder"),all=TRUE)
#all_data has 35 more observations than df (climate data with consensus >=0,8)
#meaning that some files from the header data have empty values for the climate data
all_data <- all_data[!is.na(all_data$final_value),]

#quality of header data
df <- header_data[header_data$consensus >= 0.8, ]
percentage_good_quality = length(df$value)/length(header_data$value)*100
#98,37% of the header data has a consensus >0,8

# the header observations with less than 0.6 consensus are manually checked
# since there are only 40 observations with a consensus lower than 0,6
df <- header_data[header_data$consensus < 0.6, ]
#most values have low consensus but are in fact correct (checked with images)
#or where already adjusted in previous step when creating the final dataset
#the following where indeed false and are corrected
all_data[all_data$file=="capture_72.jpg" & all_data$folder==6317, ]$month = "Avril"
all_data[all_data$file=="capture_32.jpg" & all_data$folder==7786, ]$year = "54"
all_data[all_data$file=="capture_76.jpg" & all_data$folder==7851, ]$year = "54"


# now the header data can be assumed to be of good quality

all_data <- all_data[order(all_data[,2], all_data[,1],all_data[,4],all_data[,3] ),]
table(all_data$month,useNA = "always")
table(all_data$year,useNA = "always")
table(all_data$location,useNA = "always")
#192 values of NA for both month, year and location
#remove these NA
all_data <- all_data[!is.na(all_data$month),]

#convert date info in format 'yyyy-mm-dd'
month <- names(table(all_data$month))
month <- month[c(5,4,9,2,8,7,6,1,12,11,10,3)]
month_new <- 1:12
df <- all_data
for (i in 1:length(month)){
  df$month[(df$month)==month[i]] <- month_new[i]
}

df <- df |> mutate(
  year= paste("19",year,sep=""),
  date = paste(year,month,row,sep="-"),
  date= as.Date(date),
  day= row,
  month=as.numeric(month))

all_data <- df[,c("date","file","folder","column","final_value","consensus","location","year","month","day","consensus year","consensus month")]

#sort final values by Tmax, Tmin and Prec
table(all_data$column)
#1 row with column = 5 ? doesn't belong in this dataset since this is not T or P
all_data <- all_data[!(all_data$column)=="5",]
#column = 1 = Tmax
#column = 2 = Tmin
#column = 8 = P(mm)
all_data[all_data$column=="1", ]$column = "Tmax"
all_data[all_data$column=="2", ]$column = "Tmin"
all_data[all_data$column=="8", ]$column = "Prec"

#save this table
write.table(all_data,"data/good_quality_data.csv",col.names = TRUE,
  row.names = FALSE,quote = FALSE,sep = ",")

#Information per location (station)
table(all_data$column)
length(table(all_data$folder))
length(table(all_data$location))
#there are 42 folders and 40 locations
table(all_data$location)
#Kasongo en Bondo have more than 8000 observations--> have each 2 folders 
#Kasongo: 6319 en 7294 (respectively station 13008 and 42009)
#Bondo: 6736 en 6771 (respectively station 32007 en 32026)

summary <- all_data |>
  group_by(folder,location,column) |>
  summarize( 
            mean_consensus = round(mean(consensus), digits = 2),
            first = as.Date(first(names(table(date)))),
            last= as.Date(last(names(table(date)))),
            max_year = as.numeric(names(which.max(table(year)))),
            total_observations = n(),
            possible_observations= last-first,
            perc_missing =round(((as.numeric(possible_observations)-total_observations)/
                             as.numeric(possible_observations))*100,digits=2)
            )
names(summary)[names(summary) == "column"] <- "variable"

summary[summary$perc_missing== min(summary$perc_missing),]
#Bondo: Tmax has lowest amount of missing values
mean(summary[summary$variable=="Prec",]$perc_missing) #68,37%
mean(summary[summary$variable=="Tmax",]$perc_missing) #34%
mean(summary[summary$variable=="Tmin",]$perc_missing) #34,41%

#Data for precipitation is still lacking!

max(table(all_data$folder))
#folder 7718 has 7002 observations = maximum 

write.table(summary,"data/summary_locations_good_quality.csv",col.names = TRUE,
            row.names = FALSE,quote = FALSE,sep = ",")





