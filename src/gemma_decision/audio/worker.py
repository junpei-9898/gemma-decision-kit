# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
"""Offline isolated MOSS inference, adapted from official inference_utils.py.
Source: OpenMOSS/MOSS-Transcribe-Diarize commit61bc29cd4120be7b5d3b761b64cd5dff57263642.
Changes: predecoded PCM, pinned hashes/versions, bounded complete generation,
strict parser coverage, content-free failures, structured private output.
"""
import argparse
import dataclasses
import importlib.metadata
import json
import re
import sys
import time
from .contracts import AudioError,manifest,validate_transcript
from .model import verify_model
from .moss_parser import parse_transcript

DEFAULT_PROMPT=('请将音频转写为文本，每一段需以起始时间戳和说话人编号'
                '（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，'
                '并在段末标注结束时间戳，以清晰标明该段语音范围。')


def parsed_segments(text,duration):
    segments=[dataclasses.asdict(s) for s in parse_transcript(text)]
    stripped=re.sub(r'\[(?:\d+(?:\.\d+)?|S\d+)\]','',text)
    compact=lambda s:''.join(s.split())
    if compact(stripped)!=compact(''.join(s['text'] for s in segments)):
        raise AudioError('Speech output has unparsed content; no partial transcript accepted')
    return segments



def parse_with_warnings(text,duration):
    try:return parsed_segments(text,duration), []
    except AudioError:
        # Some single-speaker generations omit all speaker tags. Preserve text/time,
        # but explicitly label identity unknown; never infer a speaker from the voice.
        pattern=re.compile(r'\s*\[(\d+(?:\.\d+)?)\]([^\[\]]+)\[(\d+(?:\.\d+)?)\]\s*')
        cursor=0;segments=[]
        for match in pattern.finditer(text):
            if match.start()!=cursor:raise AudioError('Speech output has unparsed content; no partial transcript accepted')
            segments.append({'start':float(match[1]),'end':float(match[3]),'speaker':'S0000','text':match[2].strip()})
            cursor=match.end()
        if not segments or cursor!=len(text):raise AudioError('Speech output has unparsed content; no partial transcript accepted')
        return segments,['missing_speaker_labels']


def infer(args):
    if importlib.metadata.version('transformers')!='5.8.1':raise AudioError('MOSS requires isolated Transformers5.8.1')
    import torch
    import soundfile as sf
    from transformers import AutoModelForCausalLM,AutoProcessor
    if not torch.cuda.is_available():raise AudioError('MOSS worker requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(0)
    path=verify_model(args.model)
    audio,sr=sf.read(args.audio,dtype='float32')
    if sr!=16000 or audio.ndim!=1 or not 0<len(audio)<=1800*16000:raise AudioError('Worker PCM bounds exceeded')
    duration=len(audio)/sr
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    processor=AutoProcessor.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    calls=[0]
    def observe(*unused):calls[0]+=1
    model.register_forward_pre_hook(observe)
    messages=[{'role':'user','content':[{'type':'audio','audio':'local-pcm'},{'type':'text','text':DEFAULT_PROMPT}]}]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=text,audio=[audio],max_length=131072-args.max_new_tokens,audio_kwargs={'device':'cuda'},return_tensors='pt').to('cuda')
        prompt=int(inputs['attention_mask'][0].sum().item())
        outputs=model.generate(**{k:inputs[k] for k in ['input_ids','attention_mask','input_features','audio_feature_lengths','audio_chunk_mapping']},max_new_tokens=args.max_new_tokens,do_sample=False)
    generated=outputs[0][prompt:]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
    if not len(generated) or int(generated[-1]) not in eos:raise AudioError('Speech generation incomplete or truncated')
    raw=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
    segments,parse_warnings=parse_with_warnings(raw,duration)
    torch.cuda.synchronize();pin=manifest()
    result={'schema_version':1,'status':'complete','model':{'id':pin['id'],'revision':pin['revision']},'duration_seconds':duration,'sample_rate':16000,'segments':segments,
            'warnings':['speaker_identity_unverified','utterance_timestamps_not_word_timestamps']+parse_warnings+([] if segments else ['empty_transcript_unverified']),
            'usage':{'prompt_tokens':prompt,'generated_tokens':int(generated.numel()),'forward_calls':calls[0],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'inference_seconds':time.perf_counter()-start}}
    return validate_transcript(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--max-new-tokens',type=int,default=16384);args=p.parse_args()
    try:
        if not 1<=args.max_new_tokens<=16384:raise AudioError('Invalid token limit')
        from .pipeline import write_private_json
        write_private_json(args.output,infer(args))
    except Exception as exc:
        # No paths, generated speech, traceback or input-bearing exception message.
        detail=str(exc) if isinstance(exc,AudioError) else 'MOSS runtime failed ('+type(exc).__name__+')'
        try:write_private_json(args.output+'.error.json',{'error':detail})
        except Exception:pass
        sys.stderr.write('MOSS worker failed ('+type(exc).__name__+')\n');return 2
    return 0

if __name__=='__main__':sys.exit(main())
