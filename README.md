# adversarial_testing_rl

# Adversarial Testing for HighwayEnv

  

This repository contains the code for the replication package of adversarial testing rl for HighwayEnv



## Usage

    
Step 1: Train the Ego ADS with NPCS:

Run:
```python

python hw_train_agent_1.py #Highway

python ra_train_agent_1.py #Roundabout

python mg_train_agent_1.py #Merge

```

  

Step 2: Test the Ego ADS with NPCs:

Open the respective file and replace the variable `ego`  with the path for the respective ego model (from step 1). Then, run:
```python

python hw_test_agent_1.py #Highway

python ra_test_agent_1.py #Roundabout

python mg_test_agent_1.py #Merge

```

  

Step 3: Train the Adversarial ADS:

Open the respective file and replace the variable `ego`  with the path for the respective ego model (from step 1). Then, run:

```python

python hw_train_agent_2.py #Highway

python ra_train_agent_2.py #Roundabout

python mg_train_agent_2.py #Merge

```

Step 4: Test the Adversarial ADS :

Open the respective file and
-  Replace the variable `ego`  with the path for the respective fine-tuned model (from step 1). 
-  Fill the `adv_list` with the paths for the models from step 3.
Then, run:
```python

python hw_test_agent_2.py #Highway

python ra_test_agent_2.py #Roundabout

python mg_test_agent_2.py #Merge
  ```
  

Step 5: Train the Purely Adversarial ADS:

Open the respective file and replace the variable `ego`  with the path for the respective ego model (from step 1). Then, run:

```python

python hw_train_agent_2_purely_adv.py #Highway

python ra_train_agent_2_purely_adv.py #Roundabout

python mg_train_agent_2_purely_adv.py #Merge

```


Step 6: Test the Purely Adversarial:

Open the respective file and
-  Replace the variable `ego`  with the path for the respective fine-tuned model (from step 1). 
-  Fill the `adv_list` with the paths for the models from step 5.
Then, run:
```python

python hw_test_agent_2_purely_adv.py #Highway

python ra_test_agent_2_purely_adv.py #Roundabout

python mg_test_agent_2_purely_adv.py #Merge

```

Step 8: Retrain the Ego ADS in the presence of the best adversarial:

Open the respective file and replace the variable `ego`  with the path for the respective ego model (from step 1) and the variable `adv` with the path for the best adv (from step 3) or purely adv model (from step 5) . Then, run:

```python

python hw_retrain_agent_1.py #Highway

python ra_retrain_agent_1.py #Roundabout

python mg_retrain_agent_1.py #Merge
```

Step 9: Test the retrained Ego ADSs in the presence of the best adversarial:

Open the respective file and
-  Fill the`ego_list`  with the paths for the models from step 8. 
-  Replace the `best_adv` variable with the path for for the best adv (from step 3) or purely adv model (from step 5)
Then, run:

```python

python hw_test_retrained_agent.py #Highway

python ra_test_retrained_agent.py #Roundabout

python mg_test_retrained_agent.py #Merge
```

Step 10: Replay Episode with Ego in learning mode for Avoidability Check

Open the respective file and
-  Replace the variable `ego`  with the path for the respective fine-tuned model (from step 1). 
-  Replace the variable `adv_model_path`  with the path for the respective fine-tuned model (from step 3 or 5). 
- Fill the `failures_path_list` with the file from step 4 or 6
Then, run:

```python

python hw_focus_train.py #Highway

python ra_focus_train.py #Roundabout

python mg_focus_train.py #Merge

```



 