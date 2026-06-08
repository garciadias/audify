#!/usr/bin/env python3
"""
Batch compare all processed audiobooks against their source files.
Ranks results to find the worst-performing audiobooks.
"""

import json, sys, time
from pathlib import Path

sys.path.insert(0, '/home/rd24/git/audify')
from audify.verify import verify_audiobook

OUTPUT_DIR = Path("data/output")
BOOKS_EN = Path("/home/rd24/Downloads/books_deepseek/en")
BOOKS_PT = Path("/home/rd24/Downloads/books_deepseek/pt")

PAIRS = [
    (OUTPUT_DIR / "adam_smith_in_beijing", BOOKS_EN / "Adam Smith in Beijing.epub"),
    (OUTPUT_DIR / "coding_democracy_how_hackers_are_disrupting_power_surveillance_and_authoritarianism_the_mit_press",
     BOOKS_EN / "Coding Democracy: How Hackers Are Disrupting Power, Surveillance, and Authoritarianism (The MIT Press).epub"),
    (OUTPUT_DIR / "decolonization_development_and_knowledge_in_africa_turning_over_a_new_leaf",
     BOOKS_EN / "Decolonization, Development and Knowledge in Africa: Turning Over a New Leaf.epub"),
    (OUTPUT_DIR / "global_history__a_view_from_the_south",
     BOOKS_EN / "Global History : A View From the South.epub"),
    (OUTPUT_DIR / "imperialism_the_highest_stage_of_capitalism_the_highest_stage_of_capitalism",
     BOOKS_EN / "Imperialism: The Highest Stage of Capitalism: The Highest Stage of Capitalism.epub"),
    (OUTPUT_DIR / "learning_ray__flexible_distributed_python_for_machine_learning",
     BOOKS_EN / "LEARNING RAY : flexible distributed python for machine learning.epub"),
    (OUTPUT_DIR / "qed_the_strange_theory_of_light_and_matter_princeton_science_library",
     BOOKS_EN / "QED: The Strange Theory of Light and Matter (Princeton Science Library).epub"),
    (OUTPUT_DIR / "red_swan__how_unorthodox_policy_making_facilitated_chinas_rise",
     BOOKS_EN / "Red Swan : How Unorthodox Policy Making Facilitated China's Rise.epub"),
    (OUTPUT_DIR / "rethinking_chinese_politics", BOOKS_EN / "Rethinking Chinese Politics.epub"),
    (OUTPUT_DIR / "six_easy_pieces__essentials_of_physics_explained_by_its_most_brilliant_teacher",
     BOOKS_EN / "Six easy pieces : essentials of physics, explained by its most brilliant teacher.epub"),
    (OUTPUT_DIR / "stalin_the_history_and_critique_of_a_black_legend",
     BOOKS_EN / "Stalin: The History and Critique of a Black Legend.epub"),
    (OUTPUT_DIR / "the_death_of_homo_economicus__work_debt_and_the_myth_of_endless_accumulation",
     BOOKS_EN / "The Death of Homo Economicus : Work, Debt and the Myth of Endless Accumulation.epub"),
    (OUTPUT_DIR / "the_jakarta_method__washingtons_anticommunist_crusade_and_the_mass_murder_program_that_shaped_our_world",
     BOOKS_EN / "The Jakarta Method : Washington's Anticommunist Crusade and the Mass Murder Program That Shaped Our World.epub"),
    (OUTPUT_DIR / "the_wandering_earth", BOOKS_EN / "The Wandering Earth.epub"),
    (OUTPUT_DIR / "who_paid_the_pipers_of_western_marxism",
     BOOKS_EN / "Who Paid the Pipers of Western Marxism?.epub"),
    (OUTPUT_DIR / "vidas_secas", BOOKS_PT / "Vidas secas.epub"),
    (OUTPUT_DIR / "parque_industrial", BOOKS_PT / "Parque industrial.epub"),
    (OUTPUT_DIR / "os_camisas_negras_e_a_esquerda_radical_em_portugues_do_brasil",
     BOOKS_PT / "Os Camisas Negras e a Esquerda Radical (Em Portugues do Brasil).epub"),
    (OUTPUT_DIR / "os_fuzis_e_as_flechas_historia_de_sangue_e_resistencia_indigena_na_ditadura",
     BOOKS_PT / "Os fuzis e as flechas: História de sangue e resistência indígena na ditadura.epub"),
]

def run_comparisons():
    results = []
    
    for out_dir, source_path in PAIRS:
        if not out_dir.is_dir():
            print(f"SKIP: {out_dir.name} (no output dir)")
            continue
        if not source_path.exists():
            print(f"SKIP: {out_dir.name} (no source: {source_path.name})")
            continue
        
        m4b_files = list(out_dir.glob("*.m4b"))
        if not m4b_files:
            print(f"SKIP: {out_dir.name} (no M4B)")
            continue
        m4b_path = m4b_files[0]
        
        print(f"  🔄 {out_dir.name}...", end=" ", flush=True)
        t0 = time.time()
        
        try:
            report = verify_audiobook(str(source_path), str(m4b_path))
            elapsed = time.time() - t0
            
            dur = report.duration_hint or report.analyze_duration()
            
            # Convert to dict
            data = {
                'dir_name': out_dir.name,
                'source_name': source_path.name,
                'm4b_name': m4b_path.name,
                'elapsed_seconds': round(elapsed, 1),
                'duration_ratio': round(dur.ratio, 3) if dur.ratio else None,
                'duration_actual_s': dur.actual_duration_s,
                'duration_expected_s': dur.expected_duration_s,
                'word_count': dur.source_word_count,
                'chapters_matched': report.matched,
                'chapters_total_source': report.total_source,
                'chapters_total_audio': report.total_audiobook,
                'chapters_missing': len(report.missing),
                'chapters_extra': len(report.extra),
                'order_violations': len(report.order_violations),
                'match_pct': report.overall_match_percentage,
            }
            results.append(data)
            
            print(f"✔ ratio={dur.ratio:.1%} ch={report.matched}/{report.total_source} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"✘ {type(e).__name__}: {e}")
    
    return results

def print_results(results):
    valid = [r for r in results if r.get('duration_ratio') is not None]
    valid.sort(key=lambda x: x.get('duration_ratio', 1))
    
    print("\n" + "="*100)
    print("TOP 5 WORST AUDIOBOOKS (lowest duration ratio)")
    print("="*100)
    
    for rank, r in enumerate(valid[:5], 1):
        ratio = r['duration_ratio']
        
        if ratio < 0.25:
            grade = "🔴 CRITICAL"
        elif ratio < 0.45:
            grade = "🟡 POOR"
        elif ratio < 0.60:
            grade = "🟠 ACCEPTABLE"
        else:
            grade = "🟢 GOOD"
        
        actual_m = r['duration_actual_s'] / 60
        expected_m = r['duration_expected_s'] / 60
        
        print(f"\n  #{rank}: {grade}")
        print(f"      Book:     {r['dir_name']}")
        print(f"      Source:   {r['source_name']}")
        print(f"      Ratio:    {ratio:.1%} ({actual_m:.0f}min / {expected_m:.0f}min expected)")
        print(f"      Words:    {r['word_count']:,} ({r['word_count']/75:.0f}min @75wpm)")
        print(f"      Chapters: {r['chapters_matched']}/{r['chapters_total_source']} matched, "
              f"{r['chapters_missing']} missing, {r['chapters_extra']} extra")
        if r['order_violations']:
            print(f"      Order issues: {r['order_violations']}")
    
    # Print all sorted
    print("\n" + "="*100)
    print("ALL RANKED RESULTS")
    print("="*100)
    print(f"{'Rank':<6} {'Ratio':<8} {'Chapters':<18} {'Words':<10} {'Time':<8} {'Name'}")
    print("-"*100)
    for rank, r in enumerate(valid, 1):
        ratio = r['duration_ratio']
        ch = f"{r['chapters_matched']}/{r['chapters_total_source']}"
        words = f"{r['word_count']:,}"
        elapsed = f"{r['elapsed_seconds']}s"
        name = r['dir_name'][:50]
        print(f"  {rank:<4} {ratio:<8.1%} {ch:<18} {words:<10} {elapsed:<8} {name}")
    
    return valid

# Main
results = run_comparisons()
valid = print_results(results)

with open('batch_compare_results.json', 'w') as f:
    json.dump(valid, f, indent=2)

print(f"\n✅ Saved to batch_compare_results.json ({len(valid)} comparisons)")
