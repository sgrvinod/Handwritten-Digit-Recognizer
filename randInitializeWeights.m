function W = randInitializeWeights(L_in, L_out)
%Function to randomly intialize mapping weights between -epsilon and
%epsilon

W = zeros(L_out, 1 + L_in);
epsilon=sqrt(6)/(sqrt(L_in+L_out));
W=(rand(L_out,1+L_in)*2*epsilon)-epsilon;


end
