// SPDX-License-Identifier: Apache-2.0
// Thin synchronous ABI; decision semantics are unmodified upstream modules.
#[path = "../vendor/api/protocol.rs"]
pub mod protocol;
#[path = "../vendor/api/decisions.rs"]
pub mod decisions;
use eider_runtime::{chat::CheckpointChatTemplate, decision::*};
use serde_json::{Value, json};
use std::{ffi::{CStr,CString,c_char}, sync::{Mutex,OnceLock}};
struct State { compiler: Option<DecisionPromptCompiler>, pending: Option<(String,DecisionRequest)> }
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
fn execute(v: Value) -> Result<Value,String> {
    let mut s=STATE.get_or_init(||Mutex::new(State{compiler:None,pending:None})).lock().map_err(|_|"bridge poisoned")?;
    match v["op"].as_str().ok_or("missing operation")? {
        "init" => {
            if s.pending.is_some() {return Err("pending request".into());}
            let t=CheckpointChatTemplate::from_model_dir(v["model_dir"].as_str().ok_or("missing model_dir")?).map_err(|e|e.to_string())?;
            s.compiler=Some(DecisionPromptCompiler::new(t).map_err(|e|e.to_string())?);
            Ok(json!({"prompt_format":DECISION_PROMPT_FORMAT}))
        }
        "prepare" => {
            if s.pending.is_some() {return Err("finish or discard pending request first".into());}
            let body:decisions::DecisionApiRequest=serde_json::from_value(v["body"].clone()).map_err(|e|e.to_string())?;
            let model=body.model.clone();
            let req=body.into_prompt_request().map_err(|e|format!("{e:?}"))?;
            let p=s.compiler.as_ref().ok_or("not initialized")?.prepare(req).map_err(|e|e.to_string())?;
            let result=json!({"prompt_format":p.prompt_format,"prefix_tokens":p.prefix_tokens,"logical_input_tokens":p.logical_input_tokens(),"branches":p.branches.iter().map(|b|json!({"question_id":b.question_id,"suffix_tokens":b.suffix_tokens,"label_token_ids":b.label_token_ids,"option_keys":b.option_keys})).collect::<Vec<_>>()});
            s.pending=Some((model,p));Ok(result)
        }
        "discard" => {s.pending=None;Ok(json!({}))}
        "finish" => {
            let (model,p)=s.pending.as_ref().ok_or("no pending request")?;
            let rows=v["logits"].as_array().ok_or("missing logits")?;
            if rows.len()!=p.branches.len(){return Err("wrong branch count".into());}
            let mut logits=Vec::new();
            for (row,b) in rows.iter().zip(&p.branches) {
                if row["question_id"].as_str()!=Some(&b.question_id){return Err("question order/identity mismatch".into());}
                let values:Vec<f32>=serde_json::from_value(row["values"].clone()).map_err(|e|e.to_string())?;
                if values.iter().any(|x|!x.is_finite()){return Err("nonfinite logits".into());}
                logits.push(DecisionBranchLogits{question_id:b.question_id.clone(),logits:values});
            }
            let answers=answers_from_logits(p,&logits,1.0).map_err(|e|e.to_string())?;
            let completion=DecisionCompletion{answers,usage:DecisionUsage{input_tokens:p.logical_input_tokens(),output_tokens:p.branches.len()},timings:Default::default(),released_sequence_device_bytes:0};
            let response=decisions::DecisionApiResponse::from_completion(model.clone(),completion,None).map_err(|e|format!("{e:?}"))?;
            let result=serde_json::from_str::<Value>(&serde_json::to_string(&response).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            s.pending=None;Ok(result)
        }
        _ => Err("unknown operation".into())
    }
}
/// Caller passes a valid nul-terminated UTF-8 string and frees the returned buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eider_call(input:*const c_char)->*mut c_char {
    let result=std::panic::catch_unwind(|| {
        if input.is_null(){return Err("null input".to_string());}
        let bytes=unsafe {CStr::from_ptr(input)}.to_bytes();
        let value=serde_json::from_slice(bytes).map_err(|e|e.to_string())?;
        execute(value)
    }).unwrap_or_else(|_|Err("bridge panic".into()));
    let value=match result {Ok(v)=>json!({"ok":v}),Err(e)=>json!({"error":e})};
    CString::new(value.to_string()).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eider_free(ptr:*mut c_char) {
    if !ptr.is_null(){drop(unsafe {CString::from_raw(ptr)});}
}
#[cfg(test)]
mod contract_tests;
