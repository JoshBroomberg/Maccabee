# Custom R program
library("utils")
capture.output(library("Matching"))

p_score_match <- function(Y, Tr, X){
    out <- Match(
        Y=Y,
        Tr=Tr,
        X=X,
        estimand="ATT",
        replace=TRUE,
        version="fast")
        
    return(out[["est"]][1][1])
}
