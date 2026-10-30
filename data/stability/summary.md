# Battery policy reliability (battery_tuned)

Same scenario, reward and 2M-step budget throughout; training seeds 0/1/2;
evaluated on episodes 1000-1029. 'Best checkpoint' is the checkpoint with the
highest mean reward on the training run's own evaluation environment, whose
episodes are drawn at random and are not the 1000-1029 test episodes.

| Setting | Seeds | Success /30 | Total | Mean steps | Charging steps | Outcomes |
|---|---|---|---|---|---|---|
| lr 3e-4, final policy (published + seed study) | 3 | 27 / 26 / 19 | 72/90 | 87.89 / 76.04 / 65.95 | 15.43 / 12.23 / 206.53 | all_tasks_delivered 27, battery_depleted 3 ; all_tasks_delivered 26, battery_depleted 4 ; all_tasks_delivered 19, battery_depleted 1, timeout 10 |
| lr 3e-4, best checkpoint | 3 | 30 / 27 / 25 | 82/90 | 89.37 / 75.70 / 71.44 | 15.93 / 12.63 / 10.10 | all_tasks_delivered 30 ; all_tasks_delivered 27, battery_depleted 3 ; all_tasks_delivered 25, battery_depleted 5 |
| lr 1e-4, final policy | 3 | 6 / 30 / 0 | 36/90 | 76.50 / 78.07 / n/a | 2.67 / 10.23 / 294.13 | all_tasks_delivered 6, battery_depleted 24 ; all_tasks_delivered 30 ; timeout 30 |
| lr 1e-4, best checkpoint | 3 | 5 / 30 / 0 | 35/90 | 75.80 / 79.23 / n/a | 2.57 / 10.30 / 249.90 | all_tasks_delivered 5, battery_depleted 25 ; all_tasks_delivered 30 ; timeout 30 |
