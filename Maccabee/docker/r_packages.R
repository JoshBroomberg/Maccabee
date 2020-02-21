options(repos=structure(c(CRAN="https://cloud.r-project.org/")))

pkgLoad <- function(packages) {
    packagecheck <- match( packages, utils::installed.packages()[,1] )

    packagestoinstall <- packages[ is.na( packagecheck ) ]

    if( length( packagestoinstall ) > 0L ) {
        utils::install.packages(packagestoinstall)
    } else {
        print( "All requested packages already installed" )
    }

    for( package in packages ) {
        suppressPackageStartupMessages(
            library( package, character.only = TRUE, quietly = TRUE )
        )
    }

}

pkgLoad(c())
install.packages('Matching')

#library(devtools)
#devtools::install_github("soerenkuenzel/causalToolbox")
