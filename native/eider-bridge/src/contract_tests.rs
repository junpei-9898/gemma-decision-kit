use super::*;
fn ffi(v:Value)->Value {
 let input=CString::new(v.to_string()).unwrap();
 unsafe {let ptr=eider_call(input.as_ptr());let result=serde_json::from_slice(CStr::from_ptr(ptr).to_bytes()).unwrap();eider_free(ptr);result}
}
#[test]
fn direct_upstream_equals_ffi() {
 let dir=std::env::var("EIDER_TEST_MODEL").expect("local model required");
 let cases:Vec<Value>=serde_json::from_slice(&std::fs::read(std::env::var("EIDER_TEST_CASES").unwrap()).unwrap()).unwrap();
 let compiler=DecisionPromptCompiler::new(CheckpointChatTemplate::from_model_dir(&dir).unwrap()).unwrap();
 assert!(ffi(json!({"op":"init","model_dir":dir})).get("ok").is_some());
 for c in &cases {
  let mut body=c["body"].clone();body["model"]=json!("local-gemma4");
  let request:decisions::DecisionApiRequest=serde_json::from_value(body.clone()).unwrap();
  let p=compiler.prepare(request.into_prompt_request().unwrap()).unwrap();
  let prepared=ffi(json!({"op":"prepare","body":body}));
  assert_eq!(prepared["ok"]["prefix_tokens"],json!(p.prefix_tokens));
  let mut logits=Vec::new();let mut wire=Vec::new();
  for (i,b) in p.branches.iter().enumerate() {
   for (key,value) in [("suffix_tokens",json!(b.suffix_tokens)),("label_token_ids",json!(b.label_token_ids)),("option_keys",json!(b.option_keys))] {assert_eq!(prepared["ok"]["branches"][i][key],value);}
   let values:Vec<f32>=(0..b.option_keys.len()).map(|i|if i%2==0 {-100.0+i as f32}else{17.3-i as f32}).collect();
   wire.push(json!({"question_id":b.question_id,"values":values}));
   logits.push(DecisionBranchLogits{question_id:b.question_id.clone(),logits:values});
  }
  let direct=decisions::DecisionApiResponse::from_completion("local-gemma4".into(),DecisionCompletion{answers:answers_from_logits(&p,&logits,1.0).unwrap(),usage:DecisionUsage{input_tokens:p.logical_input_tokens(),output_tokens:p.branches.len()},timings:Default::default(),released_sequence_device_bytes:0},None).unwrap();
  assert!(ffi(json!({"op":"finish","logits":[]})).get("error").is_some());
  assert_eq!(ffi(json!({"op":"finish","logits":wire}))["ok"],serde_json::from_str::<Value>(&serde_json::to_string(&direct).unwrap()).unwrap());
 }
 assert!(ffi(json!({"op":"finish","logits":[]})).get("error").is_some());
 assert!(ffi(json!({"op":"prepare","body":{"model":"x","state":"x","questions":{}}})).get("error").is_some());
 println!("{} cases direct upstream/FFI exact",cases.len());
}
