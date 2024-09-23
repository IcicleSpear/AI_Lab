# Ex.No: 4   Implementation of Alpha Beta Pruning 
### DATE: 12-09-24                                                                           
### REGISTER NUMBER : 212221040004
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

### Program:
```py
INF = float('inf')

def alpha_beta_pruning(depth, node_index, maximizing_player, values, alpha, beta, max_depth):

    if depth == max_depth:
        return values[node_index]

    if maximizing_player:
        max_eval = -INF
        for i in range(2):
            eval = alpha_beta_pruning(depth + 1, node_index * 2 + i, False, values, alpha, beta, max_depth)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = INF
        for i in range(2):
            eval = alpha_beta_pruning(depth + 1, node_index * 2 + i, True, values, alpha, beta, max_depth)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            
            if beta <= alpha:
                break
        return min_eval

if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    max_depth = 3
    print("Optimal value:", alpha_beta_pruning(0, 0, True, values, -INF, INF, max_depth))

```

### Output:
![image](https://github.com/user-attachments/assets/62b15d2e-448a-4a6e-b663-8a3fdce61b6a)

### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
