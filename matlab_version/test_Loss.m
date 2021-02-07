function test_Loss(x_test,encoder,decoder)
x_test = dlarray(x_test,'SSCB');

test_encoded = encoder.predict(x_test);

split = size(test_encoded,1)/2;

test_means = test_encoded(1:split,:);
test_vars = test_encoded(1+split:end,:);

num_vars = size(test_means);

epsilon = rand(num_vars);
sigma = exp(.5 * test_vars);

z = epsilon .* sigma + test_means;
z = reshape(z, [1,1,num_vars]);
z = dlarray(z,'SSCB');

prediction = decoder.predict(z);
prediction = dlarray(prediction,'SSCB');

recon_loss = crossentropy(prediction,x_test,'TargetCategories','independent');
kl_loss = -5e-4 * mean(1 + test_vars - test_means.^2 - exp(test_vars), 1);
loss = mean(recon_loss + kl_loss);
disp("Loss = "+ extractdata(loss));
end

