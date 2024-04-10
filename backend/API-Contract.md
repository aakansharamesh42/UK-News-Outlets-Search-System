# API Contract

# Search Engine

## GET /search
---
  Returns document IDs, given query.
* **URL Params**  
  127.0.0.1:8080
* **Data Params**  
  None
* **Headers**  
  Content-Type: application/json  
* **Success Response:**  
* **Code:** 200  
  **Content:**  
```
{
  query: <string>,
  top_k: <int>
}
```

  **Example Content:**  
```
{
  query: "Elon Musk built rocket to go to Mars",
  top_k: 10
}
```

* **Example Response**  
```
{
  "status": "success",
  "results": [100, 101, 102]
}
```
* **Error Response:**  
  * **Code:** 404  
  **Content:** `{ error : "Document doesn't exist" }`  

# Query Expansion

## GET /expansion
----
  Returns top-k alternative queries, given query.
* **URL Params**  
  127.0.0.1:8080
* **Data Params**  
  None
* **Headers**  
  Content-Type: application/json  
* **Success Response:**  
* **Code:** 200  
  **Content:**  
```
{
  query: <string>,
  top_k: <int>
}
```

  **Example Content:**  
```
{
  query: "Elon Musk built a rocket to go to Mars",
  top_k: 10
}
```

* **Example Response**  
```
{
  "status": "success",
  "results": [
    {
      "query": "Elon Musk's rocket",
      "score": 0.85
    },
    {
      "query": "Elon Musk wants to go to Mars",
      "score": 0.81
    },
    {
      "query": "Elon Musk and Mars",
      "score": 0.53
    },
    ...
  ]
}

```
* **Error Response:**  
  * **Code:** 404  
  **Content:** `{ error : "No Expanded Query" }`  
