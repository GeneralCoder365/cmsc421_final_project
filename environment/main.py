import marketEnv as me

environment = me.markEnv()
new_state, reward, done, _ = environment.step([1, 2, 0, 1, 2, 0, 0, 1, 1, 0, 0, 0, 2, 1, 2, 2, 1, 1, 1, 0, 0, 1, 0, 2])
print(reward)