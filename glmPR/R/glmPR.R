
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

    res$fitted.values <- exp(as.vector(X %*% res$coefficients))
    res$residuals <- y - res$fitted.values
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
    print(x$coefficients, digits=6)
}

summary.glmPR <- function(object, ...) {
    se <- object$stderr
    tval <- coef(object)/se

    TAB <- cbind(Estimate = coef(object),
                 StdErr = se,
                 t.value = tval,
                 p.value = 2*pt(-abs(tval), df=object$df))

    rownames(TAB) <- names(object$coefficients)
    colnames(TAB) <- c("Estimate", "StdErr", "t.value", "p.value")

    ## cf src/library/stats/R/lm.R and case with no weights and an intercept
    f <- object$fitted.values
    r <- object$residuals
    #mss <- sum((f - mean(f))^2)
    mss <- if (object$intercept) sum((f - mean(f))^2) else sum(f^2)
    rss <- sum(r^2)

    r.squared <- mss/(mss + rss)
    df.int <- if (object$intercept) 1L else 0L

    n <- length(f)
    rdf <- object$df
    adj.r.squared <- 1 - (1 - r.squared) * ((n - df.int)/rdf)

    res <- list(call = object$call,
                coefficients = TAB,
                r.squared = r.squared,
                adj.r.squared = adj.r.squared,
                sigma = sqrt(sum((object$residuals)^2)/rdf),
                df = object$df,
                residSum = summary(object$residuals, digits=5)[-4])

    class(res) <- "summary.glmPR"
    res
}

print.summary.glmPR <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\nResiduals:\n")
    print(x$residSum)
    cat("\n")

    printCoefmat(x$coefficients, P.values=TRUE, has.Pvalue=TRUE)
    digits <- max(3, getOption("digits") - 3)
    cat("\nResidual standard error: ", formatC(x$sigma, digits=digits), " on ",
        formatC(x$df), " degrees of freedom\n", sep="")
    cat("Multiple R-squared: ", formatC(x$r.squared, digits=digits),
        ",\tAdjusted R-squared: ",formatC(x$adj.r.squared, digits=digits),
        "\n", sep="")
    invisible(x)
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
        if (!is.matrix(newx)) {
            stop("The newx is not matrix!")
        }
        x <- newx
        y <- exp(as.vector(cbind (1, x) %*% coef(object)))
    }
    y
}