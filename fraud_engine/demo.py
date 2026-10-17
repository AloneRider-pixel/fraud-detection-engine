"""Single-page demo UI served at /."""

DEMO_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud Detection Engine - demo</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#111}
.note{background:#fff8e1;border:1px solid #e0c36a;padding:.7rem 1rem;border-radius:6px;font-size:.92rem}
label{display:block;margin:.5rem 0 .1rem;font-size:.85rem;color:#444}
input,select{width:100%;padding:.4rem;box-sizing:border-box}
.row{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
button{margin:.4rem .4rem 0 0;padding:.5rem .9rem;cursor:pointer}
#out{margin-top:1rem;padding:1rem;border:1px solid #ccc;border-radius:6px;min-height:3rem;white-space:pre-wrap}
.BLOCK{border-color:#c0392b!important}.ALLOW{border-color:#27ae60!important}
</style></head><body>
<h1>Fraud Detection Engine</h1>
<p class="note">Demo only. The model is trained on <b>synthetic</b> data, so scores say nothing about real fraud.
This free instance sleeps when idle, so the first request after a pause can take a while.
Velocity features use in-memory state that resets on restart.</p>
<div>
<button type="button" onclick="fill('drain')">Sample: account drain transfer</button>
<button type="button" onclick="fill('small')">Sample: small payment</button>
</div>
<form id="f">
<div class="row">
<div><label>Type</label><select name="type">
<option>PAYMENT</option><option>CASH_IN</option><option>CASH_OUT</option><option selected>TRANSFER</option><option>DEBIT</option></select></div>
<div><label>Step (hour of the month)</label><input name="step" type="number" value="10"></div>
<div><label>Amount</label><input name="amount" type="number" step="0.01" value="5000"></div>
<div><label>Origin account</label><input name="nameOrig" value="C1"></div>
<div><label>Origin balance before</label><input name="oldbalanceOrg" type="number" step="0.01" value="5000"></div>
<div><label>Origin balance after</label><input name="newbalanceOrig" type="number" step="0.01" value="0"></div>
<div><label>Destination account</label><input name="nameDest" value="C9"></div>
<div><label>Destination balance before</label><input name="oldbalanceDest" type="number" step="0.01" value="0"></div>
<div><label>Destination balance after</label><input name="newbalanceDest" type="number" step="0.01" value="5000"></div>
</div>
<button type="submit">Score transaction</button>
</form>
<div id="out">No result yet.</div>
<p><a href="/docs">API docs</a> - <a href="https://github.com/AloneRider-pixel/fraud-detection-engine">Source</a></p>
<script>
const S={drain:{type:"TRANSFER",step:10,amount:5000,nameOrig:"C1",oldbalanceOrg:5000,newbalanceOrig:0,nameDest:"C9",oldbalanceDest:0,newbalanceDest:5000},
small:{type:"PAYMENT",step:10,amount:40,nameOrig:"C2",oldbalanceOrg:900,newbalanceOrig:860,nameDest:"C8",oldbalanceDest:300,newbalanceDest:300}};
const f=document.getElementById("f"),out=document.getElementById("out");
function fill(k){for(const [n,v] of Object.entries(S[k]))f.elements[n].value=v;}
f.addEventListener("submit",async e=>{e.preventDefault();
const d={};for(const el of f.elements){if(!el.name)continue;d[el.name]=el.type==="number"?Number(el.value):el.value;}
out.textContent="Scoring...";out.className="";
try{const r=await fetch("/score",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(d)});
const j=await r.json();if(!r.ok){out.textContent="Error: "+JSON.stringify(j);return;}
out.className=j.action;
out.textContent="fraud_prob: "+j.fraud_prob+"\\naction: "+j.action+"\\nserver latency: "+j.latency_ms+" ms\\n(round trip includes network and any cold start)";
}catch(err){out.textContent="Request failed: "+err;}});
</script></body></html>"""
