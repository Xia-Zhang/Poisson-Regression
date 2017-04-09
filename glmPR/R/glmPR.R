
glmPREasyForm <- function(X, y, lambda, threads) {
    y_nrow <- nrow(y)
    if (is.null(y_nrow)) {
        y_nrow = length(y)
    }
    x_nrow <- nrow(X)
    if (is.null(x_nrow)) {
        x_nrow = length(X)
    }
    stopifnot(is.matrix(X), is.numeric(y), y_nrow==x_nrow)
    .Call('glmPR_glmPR', PACKAGE = 'glmPR', X, y, lambda, threads)
}

glmPRWork <- function(X, y, lambda, threads) {
    X <- as.matrix(X)
    y <- as.numeric(y)

    res <- glmPREasyForm(X, y, lambda, threads)

    res$coefficients <- as.vector(res$coefficient)
    names(res$coefficients) <- colnames(X)
    res$call <- match.call()
    res$intercept <- any(apply(X, 2, function(x) all(x == x[1])))

    class(res) <- "glmPR"
    res
}

#' L1-penalized Poisson Regression
#' You can see glmPR.default and glmPR.formula for more details
#'
#' @param X the input matrix
#' @param ... other parameter
#'
#' @return coefficients vector
#'
glmPR <- function(X, ...) UseMethod("glmPR")

#' L1-penalized Poisson Regression default function
#'
#' @param X the input matrix
#' @param y the response vector
#' @param lambda a constant scalar parameter to control the influence of L1-Norm
#' @param threads the parallelize node number
#' @param ... other parameter
#'
#' @return coefficients vector
#' @examples
#' x <- matrix(rnorm(100), ncol = 4)
#' y <- rpois(25, 3)
#' glmPR(x, y, 1.0, 4)
#'
glmPR.default <- function(X, y, lambda = 1.0, threads = 4, ...) {
    X <- as.matrix(X)
    X <- cbind(1, X)
    glmPRWork(X, y, lambda, threads)
}


print.glmPR <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\nCoefficients:\n")
    print(round(x$coefficients, 6))
}

#' L1-penalized Poisson Regression input formula
#'
#' @param formula the formula object
#' @param data the data set
#' @param lambda a constant scalar parameter to control the influence of L1-Norm
#' @param threads the parallelize node number
#' @param ... other parameter
#'
#' @return coefficients vector
#' @examples
#' x <- matrix(rnorm(100), ncol = 4)
#' y <- rpois(25, 3)
#' glmPR(y ~ x, 1.0, 4)
#'

glmPR.formula <- function(formula, data = list(), lambda = 1.0, threads = 4, ...) {
    mf <- model.frame(formula = formula, data = data)
    x <- model.matrix(attr(mf, "terms"), data = mf)
    y <- model.response(mf)

    res <- glmPRWork(x, y, lambda, threads, ...)
    res$call <- match.call()
    res$formula <- formula
    res$intercept <- attr(attr(mf, "terms"), "intercept")
    res
}


#' The predict funtion with new data
#'
#' @param object the learned glmPR object
#' @param newx the new itest data
#' @param ... other parameter
#'
#' @return the predicted vector
#' @examples
#' x <- matrix(rnorm(100), ncol = 4)
#' y <- rpois(25, 3)
#' fit <- glmPR(x, y)
#' predict(fit, x[1:3,])
#' 

predict.glmPR <- function(object, newx = NULL, ...) {
    if (is.null(newx)) {
        y <- fitted(object)
    } else {
        if (is.vector(newx)) {
            newx <- matrix(newx, nrow = 1)
        }
        if (!is.matrix(newx)) {
            stop("The newx is not matrix!")
        }
        x <- newx
        y <- exp(as.vector(cbind (1, x) %*% coef(object)))
    }
    y
}