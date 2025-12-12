#!/bin/bash
echo "=== Grid Search Status ==="
wc -l /home/jnaid/make_ai_robot/results/hyperparameter_search.log
echo ""
echo "=== Latest ATE Results ==="
grep "ATE=" /home/jnaid/make_ai_robot/results/hyperparameter_search.log | tail -10
echo ""
echo "=== Checking if still running ==="
if pgrep -f "hyperparameter_search.py" > /dev/null; then
  echo "Still running (background process active)"
else
  echo "Completed (or paused)"
fi
