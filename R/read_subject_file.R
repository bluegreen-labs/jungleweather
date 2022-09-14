#' Read subject set file
#' 
#' Used in screening the manifests
#' for already uploaded samples (if the
#' upload routine fails badly).
#'
#' @param path path of a subject set file
#' @param subject_set subject set to use
#'
#' @return
#' @export

read_subject_file <- function(
  path = "~/Desktop/jungle-weather-subjects.csv",
  subject_set = 81979
){
  df <- read.table(path,
                   header = TRUE,
                   sep = ",",
                   stringsAsFactors = FALSE)
  df <- df$metadata[df$subject_set_id == subject_set]
  df <- unlist(lapply(df, function(x){
    f <- strsplit(x,",")[[1]][1]
    f <- strsplit(f, ":")[[1]][2]
    return(gsub("\\\"","",f))
  }))  
  return(df)
}