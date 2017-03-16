
glmPREasyForm <- function(X, y, s = 0.0) {
    stopifnot(is.matrix(X), is.numeric(y), nrow(y)==nrow(X))
    .Call('glmPR_glmPR', PACKAGE = 'glmPR', X, y, s)
}

glmPRWork <- function(X, y, s) {
    X <- as.matrix(X)
    y <- as.numeric(y)

    res <- glmPREasyForm(X, y, s)

    res$coefficients <- as.vector(res$coefficient)
    names(res$coefficients) <- colnames(X)
    res$call <- match.call()
    res$intercept <- any(apply(X, 2, function(x) all(x == x[1])))

    class(res) <- "glmPR"
    res
}

glmPR <- function(X, ...) UseMethod("glmPR")

glmPR.default <- function(X, y, s = 0.0, ...) {
    X <- cbind(1, X)
    glmPRWork(X, y, s)
}

# Use the form of RcppArmadillo fastLM to present the print and summary of glmPR class.

print.glmPR <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\nCoefficients:\n")
    print(round(x$coefficients, 6))
}

glmPR.formula <- function(formula, data = list(), s = 0.0, ...) {
    mf <- model.frame(formula = formula, data = data)
    x <- model.matrix(attr(mf, "terms"), data = mf)
    y <- model.response(mf)

    res <- glmPRWork(x, y, s, ...)
    res$call <- match.call()
    res$formula <- formula
    res$intercept <- attr(attr(mf, "terms"), "intercept")
    res
}

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