function test_Seq(x_test, encoder, decoder)
% Convert sequence to dlarray
x_test = dlarray(x_test,'SSCB');

% Print the input sequence
x_test_array = extractdata(x_test);
x_test_array = x_test_array.';
disp("----------------");
disp("Input sequence: ");
disp(" ");
disp(x_test_array);
disp("----------------");
test_encoded = encoder.predict(x_test);

split = size(test_encoded,1)/2;

test_means = test_encoded(1:split,:);
test_vars = test_encoded(1+split:end,:);

num_vars = size(test_means);

% Print the means and variances of the latent variables
disp("----------------");
disp("Means: ");
disp(" ");
display_means = extractdata(test_means);
display_means = display_means.';
disp(display_means);
disp("----------------");
disp("Variances: ");
disp(" ");
display_vars = extractdata(test_vars);
display_vars = display_vars.';
disp(display_vars);

disp("----------------");
epsilon = rand(num_vars);
sigma = exp(.5 * test_vars);

z = epsilon .* sigma + test_means;
z = reshape(z, [1,1,num_vars]);
z = dlarray(z,'SSCB');
% Print the latent representation of input
disp("Z: ");
disp(" ");
display_z = extractdata(z);
display_z = permute(display_z,[1 3 2]);
display_z = reshape(display_z,[],size(display_z,2),1);
disp(display_z);

disp("----------------");
prediction = decoder.predict(z);

% Print the prediction
disp("----------------");
disp("Output sequence: ");
disp(" ");
prediction = extractdata(prediction);
prediction = prediction.';
disp(round(prediction));

disp("----------------");
end
