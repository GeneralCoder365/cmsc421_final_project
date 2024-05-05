import marketEnv as me
import lstm_agent_helper as LSTM

environment = me.markEnv(train_test_stop=0.9)
train_test = environment.get_train_test_stop()
lstm = LSTM.agent_helper(environment, environment.get_unmodified_stocks(), train_test)

done = False
new_state = 0

while not done:

    actions = lstm.action(new_state)
    new_state, reward, done, _ = environment.step(actions)
    print(reward, '\n\n')

print('Profits:', environment.get_profits(), '\n\n')
correct_decisions, num_days = lstm.get_total_correct_predictions()
print('Correct Decision:', correct_decisions, '\n\n')
percentage_correct = [x/num_days for x in correct_decisions]
print('Percent Correct:', percentage_correct, '\n\n')

results = environment.get_results()
# print(results.to_string())

results.to_csv('lstm_results.csv', index=False)