'''
Continue to test the chosen classifier model against a drifting dataset unti the returned score
falls below the threshold. The "score" is the mean accuracy on the given test data and labels.
'''
import clf_test

threshold = 0.65 # the minimum acceptable score
score = 1.0      # initialize
i_trial = 1      # initialize
while score > threshold:
    score = clf_test.test('SVC', i_trial)
    i_trial += 1
print("this model's score is now less than", str(threshold))
