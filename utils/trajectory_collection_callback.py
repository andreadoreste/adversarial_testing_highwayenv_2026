import csv

obs_vector = []
dimension_obs_vector = 30 * 2 * 2 # 30 steps, 2 vehicles, 2 positions variables (x,y)

def trajectory_collection_callback(episode, env, transition, directory, num_episodes):

    obs, reward, terminated, truncated, info = transition

    for o in obs:
        x = o[0][1]
        y = o[0][2]
        obs_vector.extend([x,y])
    
    if truncated == True or terminated == True:

        zero_quantity = dimension_obs_vector - len(obs_vector)
        for i in range(zero_quantity):
            obs_vector.append(0)
        
        if not(info["crashed"]):
            obs_vector.append("True")
        else:
            obs_vector.append("False")

        obs_vector.append(episode)
        obs_vector.append(env.seed)

        file_name = f"{str(directory)}.csv"

        with open(file_name, "a") as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(obs_vector)
        
        obs_vector.clear()

        
