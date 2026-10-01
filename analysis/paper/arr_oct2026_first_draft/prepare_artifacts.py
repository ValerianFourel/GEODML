"""Reproduce manuscript-only summaries and figures from pinned local evidence.

No model loading, inference, network requests, or ranking analysis is performed.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPECTED = {
    'compliant-candidates.jsonl': '1718321f8fc86f63d00aab30e87991ada9b62a59c5ef4ce99b5acef0df4d32e9',
    'final-axis-map.jsonl': '43189f68bcafc77f9dceb7a1a8d993251d4c2a739b401ef4fd24cb64e292682e',
}

def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def source_functions(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    selected = [ast.get_source_segment(source, n) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if len(selected) != len(names):
        raise ValueError(f'Missing function in {path}: {names}')
    return '\n\n\n'.join(selected) + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population-root', type=Path, required=True)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--example', type=Path, required=True)
    args = parser.parse_args()
    ev = HERE / 'evidence'
    app = HERE / 'appendices'
    for name, sha in EXPECTED.items():
        path = args.population_root / 'final-audit' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f'Population hash mismatch: {name}')
    prompts = read_rows(args.population_root / 'final-audit/compliant-candidates.jsonl')
    axes = read_rows(args.population_root / 'final-audit/final-axis-map.jsonl')
    by_id = {p['candidate_id']: p for p in prompts}
    assert len(prompts) == len(by_id) == len(axes) == 26009
    for a in axes:
        p = by_id[a['candidate_id']]
        assert a['text_sha256'] == p['question_sha256'] == hashlib.sha256(p['question'].encode()).hexdigest()
    assert len({p['question'] for p in prompts}) == 26009
    counts = Counter(p['keyword_id'] for p in prompts)
    stats = {'input_sha256': EXPECTED, 'rows': len(prompts), 'unique_strings': len(by_id),
             'topic_ids': len(counts), 'min_per_topic': min(counts.values()),
             'max_per_topic': max(counts.values()),
             'topic_size_distribution': dict(sorted(Counter(counts.values()).items())),
             'prompt_generators': dict(Counter(p['generator_id'] for p in prompts)),
             'exact_text_axis_joins': len(axes), 'full_design_cells_per_generator': len(prompts)*12,
             'target_slot_coverage': len(prompts)/30330,
             'scope': 'Exact final text/axis files; no population ranking results or inference.'}
    save_json(ev / 'population-counts.json', stats)
    ex = json.loads(args.example.read_text())
    ids = {s['judge_task_id'] for s in ex['cell']['sources']} | {ex['cell']['j1_task_id']}
    ex['tasks'] = {k: v for k,v in ex['tasks'].items() if k in ids}
    ex['results'] = {k: v for k,v in ex['results'].items() if k in ids}
    save_json(ev / 'running-example.json', ex)
    assert len(ex['cell']['sources']) == 3
    assert [ex['results'][s['judge_task_id']]['parsed_output']['importance'] for s in ex['cell']['sources']] == [0,0,0]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7,3.3), constrained_layout=True)
    target = [by_id[a['candidate_id']]['target_normalized_axis_1'] for a in axes]
    measured = [a['consensus_axis_1_z'] for a in axes]
    hexes = ax.hexbin(target, measured, gridsize=(47,23), mincnt=1, bins='log', cmap='Blues', linewidths=.2)
    cb = fig.colorbar(hexes, ax=ax, pad=.02)
    cb.set_label('Retained prompt strings per bin')
    for label, item, offset in zip('ABC',ex['variants'],[(12,8),(-22,-22),(-28,8)]):
        xy=(item['prompt']['target_normalized_axis_1'], item['axis']['consensus_axis_1_z'])
        ax.scatter(*xy, marker='s', s=45, color='#9c2f21', edgecolors='white', linewidths=.7, zorder=4)
        ax.annotate(label, xy, xytext=offset, textcoords='offset points', fontsize=12, fontweight='bold', color='#9c2f21',
                    arrowprops={'arrowstyle':'-', 'color':'#9c2f21'})
    ax.set(xlabel='Intended generation target (0–1)', ylabel='Measured consensus axis 1 (z units)', xlim=(-.015,1.015))
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'figures/target-measurement.pdf')
    save_json(ev/'figure-bins.json', {'population_n':26009, 'centres':hexes.get_offsets().tolist(), 'counts':hexes.get_array().tolist(),
                                    'note':'Hexagonal histogram of intended target versus measured consensus z; logarithmic colour scale.'})
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(12,4.8))
    ax.set(xlim=(0,12),ylim=(0,4.8)); ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.06',facecolor=color,edgecolor='#425569',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=15)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,color='#425569',linewidth=1.4))
    box(.15,1.7,2,1.05,'Prompt text\n(topic family)','#edf2f6')
    box(3.05,3.5,3.7,.95,'External prompt embeddings\nMeasured semantic coordinates','#e0eaf7')
    box(3.05,1.65,3.7,1.25,'Queries → frozen retrieval\n→ evidence selection\n→ generator','#e4efe9')
    box(3.05,.1,3.7,.9,'Answer + emitted ordering\n+ exact evidence records','#e4efe9')
    box(8,3.15,3.8,1.1,'Request-relevance judge\nRequest + all evidence\n→ strict ordering','#fff1dd')
    box(8,1.15,3.8,1.5,'Answer-support judge (SI-v4)\nRequest + answer → map\nMap + one source at a time\n→ tied grades / zero / missing','#fff1dd')
    arrow(2.15,2.48,3.05,3.98); arrow(2.15,2.1,3.05,2.1)
    arrow(4.9,1.65,4.9,1.0)
    arrow(6.75,2.55,8,3.55); arrow(6.75,.7,8,1.55)
    ax.text(8, .4,'Compare rankings on matched source sets;\nmeasure membership changes separately.',fontsize=14,va='center')
    fig.tight_layout(pad=.3)
    fig.savefig(HERE/'figures/study-overview.pdf',bbox_inches='tight')
    plt.close(fig)

    pipeline=REPO/'analysis/interpretability/pipeline'
    si_source=(pipeline/'source_importance_v4.py').read_text()
    for node in ast.parse(si_source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('MAP_INSTRUCTIONS','SOURCE_INSTRUCTIONS'):
            fname={'MAP_INSTRUCTIONS':'si-v4-map.txt','SOURCE_INSTRUCTIONS':'si-v4-source.txt'}[node.targets[0].id]
            (app/fname).write_text(ast.literal_eval(node.value)+'\n')
    gen_source=(pipeline/'readiness_prompt_population.py').read_text()
    for n in ast.parse(gen_source).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('render_generation_request','render_search_validation_request'):
            templates=[]
            for r in ast.walk(n):
                if isinstance(r,ast.Return) and isinstance(r.value,ast.JoinedStr):
                    chunks=[]
                    for v in r.value.values:
                        chunks.append(v.value if isinstance(v,ast.Constant) else '{'+ast.unparse(v.value)+'}')
                    templates.append(''.join(chunks))
            selected=next(t for t in templates if 'search-trigger-v2' in t or t.startswith('Independently evaluate whether the candidate is a useful online-search trigger.'))
            fname='population-generation.txt' if n.name=='render_generation_request' else 'population-validation.txt'
            (app/fname).write_text(selected+'\n')
    names=['render_generation_request','render_search_validation_request','_continuous_axis_instruction',
           '_surface_realization_instruction','_search_trigger_surface_instruction','_high_axis_action_control',
           '_axis_1_instruction','_axis_2_instruction']
    (app/'population-renderers.py').write_text('# Verbatim renderer excerpts; not a standalone executable.\n'+source_functions(pipeline/'readiness_prompt_population.py',names))
    (app/'search-renderers.py').write_text('# Verbatim renderer excerpts; constants documented in main.tex.\n'+source_functions(pipeline/'agentic_search.py',
        ['_parallel_query_prompt','_final_prompt','_reactive_prompt','_forced_finish_prompt','_evidence_index','_evidence_records','_ranking_schema','_final_schema','_action_schema']))
    (app/'relevance-renderer.py').write_text(source_functions(pipeline/'agentic_judging.py',['render_relevance_prompt','relevance_schema']))
    (app/'si-v4-schemas.py').write_text(source_functions(pipeline/'source_importance_v4.py',['_object','_array','_enum','span_schema','map_schema','source_schema','_answer_block','_item']))
    shutil.copyfile(REPO/'analysis/config/si_v4_gemma.template.json',app/'si-v4-development-config.json')
    # Extract prompt bodies by executing only pure rendering functions, not a model.
    render_ns={'Sequence':list, 'Snippet':object, 'FINAL_ANSWER_MAX_CHARACTERS':1200,'REACTIVE_MAX_ITERATIONS':3,
               'json':json,'_evidence_records':lambda _: '<evidence_records_json>'}
    for n in ast.parse((pipeline/'agentic_search.py').read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name in ['_parallel_query_prompt','_final_prompt','_reactive_prompt','_forced_finish_prompt']:
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<pure renderer>','exec'),render_ns)
    for name, fn, call_args in [('parallel-query','_parallel_query_prompt',('<request>',)),
          ('search-final','_final_prompt',('<request>',[])), ('reactive','_reactive_prompt',('<request>',[],'<iteration>')),
          ('forced-finish','_forced_finish_prompt',('<request>',[]))]:
        body=render_ns[fn](*call_args).replace('"<evidence_records_json>"','<evidence_records_json>')
        (app/(name+'.txt')).write_text(body+'\n')
    relevance_source=(pipeline/'agentic_judging.py').read_text()
    ns={'Sequence':list,'ClaimEvidence':object}
    for n in ast.parse(relevance_source).body:
        if isinstance(n,ast.FunctionDef) and n.name=='render_relevance_prompt':
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<pure relevance renderer>','exec'),ns)
    from types import SimpleNamespace
    body=ns['render_relevance_prompt'](prompt_text='<request>',evidence=[SimpleNamespace(evidence_id='<evidence_id>',title='<title>',url='<url>',text='<snippet>')])
    (app/'relevance.txt').write_text(body+'\n')

    chosen=['P006','P024','P041','P044','P068','P072','P074','P083','A002','A006']
    verified_dois = {d['key']: d for d in json.loads((ev/'doi-verification.json').read_text()) if not d.get('error')} if (ev/'doi-verification.json').exists() else {}
    bib=[]; sources=[]
    for key in chosen:
        d=json.loads((args.library_root/key/'record.json').read_text())
        fields={'author':' and '.join(d['authors']), 'title':'{'+d['title']+'}', 'year':str(d['year'])}
        entry_type = 'misc'
        if d.get('doi'):
            fields['doi']=d['doi']; fields['url']='https://doi.org/'+d['doi']
            v=verified_dois.get(key)
            if v:
                entry_type = 'article' if v['type']=='journal-article' else 'inproceedings'
                fields['journal' if entry_type=='article' else 'booktitle']=v['container'][0]
                for field in ['volume','page','publisher']:
                    if v.get(field): fields['pages' if field=='page' else field]=v[field].replace('-', '--') if field=='page' else v[field]
                if v.get('issue'): fields['number']=v['issue']
            elif key=='P044':
                entry_type='inproceedings'
                fields['booktitle']='Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining'
                fields['publisher']='Association for Computing Machinery'
                # Venue/DOI/year verified on the locally downloaded paper's first page.

        else:
            arxiv=d['source_url'].split('/')[-1]
            fields.update(eprint=arxiv,archivePrefix='arXiv',url=d['source_url'])
        bib.append('@'+entry_type+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n')
        sources.append({k:d.get(k) for k in ('key','title','authors','year','doi','source_url','abstract_source_url','pdf_identity_check')})
    (HERE/'references.bib').write_text('\n'.join(bib))
    save_json(ev/'citation-metadata.json',sources)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
