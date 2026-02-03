Coursework for EEN245 (Graph Machine Learning), Jan 2026 at Chalmers Technology University, Gothenburg, Sweden (as part of my semester exchange).

I implemented and compared graph embedding methods from scratch, focusing on how structure and random walks shape representations.

Contents:
	•	Structural node 
Degree, centralities, PageRank, clustering → PCA → node classification (Karate Club)
	•	Node2Vec (implemented, not imported)
Biased random walks (BFS vs DFS), skip-gram training, geometric shape analysis
	•	Struct2Vec-style embeddings
k-hop structural signatures → similarity graph → walks → skip-gram
	•	Graph-based sentiment analysis
Word co-occurrence graph (PPMI) → Node2Vec → document embeddings → ~81% IMDb accuracy

Takeaways:
Node2Vec captures proximity, Struct2Vec captures roles, and graph embeddings work beyond graphs.
