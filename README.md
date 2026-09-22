uvicorn app.main:app --host 127.0.0.1 --port 5000



Invoke-RestMethod `

&#x20; -Uri "http://127.0.0.1:5000/predict" `

&#x20; -Method Post `

&#x20; -ContentType "application/json" `

&#x20; -Body '{"text":"I love machine learning!"}'



