function loss = binarycrossentropy(Y,pairLabels)
    % binarycrossentropy accepts the network's prediction, Y, the true
    % label, pairLabels, and returns the binary cross-entropy loss value.
    
    % Get the precision of the prediction to prevent errors due to floating
    % point precision    
    y = extractdata(Y);
    if(isa(y,'gpuArray'))
        precision = classUnderlying(y);
    else
        precision = class(y);
    end
      
    % Convert values less than floating point precision to eps.
    Y(Y < eps(precision)) = eps(precision);
    %convert values between 1-eps and 1 to 1-eps.
    Y(Y > 1 - eps(precision)) = 1 - eps(precision);
    
    % Calculate binary cross-entropy loss for each pair
    loss = -pairLabels.*log(Y) - (1 - pairLabels).*log(1 - Y);
    
    % Sum over all pairs in minibatch and normalize.
    loss = sum(loss)/numel(pairLabels);
end