
#' Validate manifest corrections
#' 
#' visually check corrected / added files
#' to manifest statements
#'
#' @param files CSV with files to check
#' @param path location of the image (cell) data
#'
#' @return
#' @export

validate_corrections <- function(
  files = "data-raw/format_1/format_1_batch_1_corrections.csv",
  path = "/backup/cobecore/zooniverse/format_1_batch_1/"
){
  library(raster, quietly = TRUE)
  
  df <- read.table(files,
                   header = FALSE,
                   stringsAsFactors = FALSE)
  
  df <- basename(df$V1)
  
  previews <- list.files(path,
                         "*preview.jpg",
                         full.names = TRUE,
                         recursive = TRUE)

  lapply(df, function(file){
    print(file)
    img_loc <- grep(sprintf("*%s_preview.jpg",
                        tools::file_path_sans_ext(file)), previews)
    print(img_loc)
    r <- raster::brick(previews[img_loc])
    plotRGB(r)
  })
}
