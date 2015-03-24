function f = log_loss(y, y_pred)

vals = y.*log(y_pred) + (1-y).*log(1-y_pred);
f = -1*mean(vals);

end