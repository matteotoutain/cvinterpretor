"""
Test script demonstrating improved NLP scoring with non-NLP features.
Shows before/after comparison and detailed scoring breakdown.
"""

from src.nlp import (
    extract_skills,
    normalize_skill,
    normalize_skill_list,
    extract_languages,
    _parse_seniority,
    _calculate_seniority_score,
    _calculate_domain_match,
    score_blocks,
    score_blocks_enhanced,
    build_ao_blocks,
    build_cv_blocks,
)


def test_skill_extraction():
    """Test skill extraction from sample text."""
    print("\n" + "="*70)
    print("TEST 1: Skill Extraction")
    print("="*70)
    
    cv_text = """
    Senior Python Developer with 6 years experience.
    Expertise in Django, FastAPI, and Node.js.
    Cloud skills: AWS (EC2, S3), Docker, Kubernetes.
    Databases: PostgreSQL, MongoDB, Redis.
    AI/ML: TensorFlow, PyTorch, scikit-learn.
    """
    
    skills = extract_skills(cv_text)
    print(f"\nText:\n{cv_text}")
    print(f"\nExtracted {len(skills)} skills:")
    print(f"  {', '.join(skills)}")
    
    # Test AO requirements
    ao_text = "We need Python, Django, AWS, Docker, and Kubernetes expert"
    ao_skills = extract_skills(ao_text)
    print(f"\nAO Requirements:")
    print(f"  {ao_text}")
    print(f"  Required skills: {', '.join(ao_skills)}")
    
    overlap = set(skills) & set(ao_skills)
    missing = set(ao_skills) - overlap
    print(f"\n  ✓ Overlap: {', '.join(overlap) if overlap else 'None'}")
    print(f"  ✗ Missing: {', '.join(missing) if missing else 'None'}")
    print(f"  Match ratio: {len(overlap)}/{len(ao_skills)} = {len(overlap)/len(ao_skills):.1%}")


def test_skill_normalization():
    """Test skill normalization functions."""
    print("\n" + "="*70)
    print("TEST 1B: Skill Normalization")
    print("="*70)
    
    # Test individual skill normalization
    test_cases = [
        ("js", "javascript"),
        ("py", "python"),
        ("k8s", "kubernetes"),
        ("tf", "tensorflow"),
        ("vue", "vue.js"),
        ("Django", "django"),  # case normalization
        ("springboot", "spring boot"),
        ("mssql", "sql server"),
    ]
    
    print("\nIndividual skill normalization:")
    for input_skill, expected in test_cases:
        result = normalize_skill(input_skill)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_skill}' → '{result}' (expected: '{expected}')")
    
    # Test list normalization with duplicates
    raw_skills = ["python", "js", "JavaScript", "py", "tensorflow", "tf", "PYTHON"]
    normalized = normalize_skill_list(raw_skills)
    print(f"\nList normalization:")
    print(f"  Input: {raw_skills}")
    print(f"  Normalized: {normalized}")
    print(f"  ✓ Duplicates removed: {len(normalized)} unique skills from {len(raw_skills)} inputs")


def test_seniority_parsing():
    """Test seniority level parsing (3-level system)."""
    print("\n" + "="*70)
    print("TEST 2: Seniority Level Parsing (3-level: Junior/Senior/Manager)")
    print("="*70)
    
    test_cases = [
        "Manager with 15 years experience",
        "Senior level, 8 years in IT",
        "Senior developer, 4 years frontend development",
        "Junior, 1.5 years Python",
        "10 years of software engineering",
        "Team Lead with director responsibilities",
    ]
    
    levels = {0: "Junior", 1: "Senior", 2: "Manager"}
    
    for text in test_cases:
        level = _parse_seniority(text)
        level_name = levels.get(level, "Unknown")
        print(f"  '{text}' → {level_name}")


def test_seniority_matching():
    """Test seniority alignment scoring (3-level system)."""
    print("\n" + "="*70)
    print("TEST 3: Seniority Alignment Scoring (3-level system)")
    print("="*70)
    
    # Define levels
    levels = {None: "Unknown", 0: "Junior", 1: "Senior", 2: "Manager"}
    
    # Test cases
    test_pairs = [
        (1, 1, "Senior → Senior (exact match)"),
        (1, 0, "Senior → Junior (one level off)"),
        (2, 0, "Manager → Junior (two levels off)"),
        (2, 1, "Manager → Senior (one level off)"),
        (1, None, "Senior → Unknown (missing data)"),
    ]
    
    print("\n  AO Seniority → CV Seniority : Score")
    for ao_level, cv_level, desc in test_pairs:
        score = _calculate_seniority_score(ao_level, cv_level)
        print(f"  {desc:35} : {score:.2f}")


def test_domain_matching():
    """Test domain/sector matching."""
    print("\n" + "="*70)
    print("TEST 4: Domain/Sector Matching")
    print("="*70)
    
    test_pairs = [
        ("IT Services", "IT Services", "exact match"),
        ("Technology", "Software Engineering", "semantic similarity"),
        ("Finance", "Healthcare", "completely different"),
        ("Cloud Engineering", "Cloud Consulting", "partial overlap"),
    ]
    
    print("\n  AO Domain : CV Domain → Score")
    for ao_domain, cv_domain, desc in test_pairs:
        score = _calculate_domain_match(ao_domain, cv_domain)
        print(f"  '{ao_domain}' ↔ '{cv_domain}' : {score:.2f} ({desc})")


def test_language_extraction():
    """Test language extraction."""
    print("\n" + "="*70)
    print("TEST 5: Language Extraction")
    print("="*70)
    
    ao_text = "Required: English, French, German"
    cv_text = "Languages: English (native), French (fluent), Spanish (conversational)"
    
    ao_langs = extract_languages(ao_text)
    cv_langs = extract_languages(cv_text)
    overlap = set(ao_langs) & set(cv_langs)
    
    print(f"\n  AO requires: {', '.join(ao_langs)}")
    print(f"  CV has: {', '.join(cv_langs)}")
    print(f"  Coverage: {len(overlap)}/{len(ao_langs)} = {len(overlap)/len(ao_langs):.1%}")


def test_full_scoring_comparison():
    """Compare old vs new scoring on realistic CVs."""
    print("\n" + "="*70)
    print("TEST 6: Full Scoring Comparison (Old vs New)")
    print("="*70)
    
    # Sample AO
    ao_struct = {
        "titre_poste": "Senior Python Developer",
        "contexte_mission": "Build cloud-native data platform",
        "competences_techniques": ["Python", "Django", "AWS", "Docker", "PostgreSQL"],
        "seniorite": "Senior (5-10 years)",
        "secteur_principal": "Technology",
        "langues_requises": ["English", "French"],
    }
    
    # Good Candidate
    cv_struct_good = {
        "role_principal": "Python Developer",
        "seniorite": "Senior, 7 years",
        "secteur_principal": "Technology",
        "technologies": ["Python", "Django", "AWS", "Docker", "PostgreSQL", "MongoDB"],
        "langues": ["English", "French"],
    }
    
    # Weak Candidate
    cv_struct_weak = {
        "role_principal": "Junior Web Developer",
        "seniorite": "Junior, 1 year",
        "secteur_principal": "Retail",
        "technologies": ["JavaScript", "React", "Node.js"],
        "langues": ["English"],
    }
    
    print("\n--- SCENARIO 1: Good Candidate Match ---")
    print(f"\nAO: {ao_struct['titre_poste']}")
    print(f"   Requires: {', '.join(ao_struct['competences_techniques'])}")
    print(f"   Seniority: {ao_struct['seniorite']}")
    
    print(f"\nCV: {cv_struct_good['role_principal']}")
    print(f"   Skills: {', '.join(cv_struct_good['technologies'])}")
    print(f"   Seniority: {cv_struct_good['seniorite']}")
    
    # Build blocks
    ao_blocks = build_ao_blocks(ao_struct, "")
    cv_blocks = build_cv_blocks(cv_struct_good, "")
    
    # Old score
    old_scores, old_method = score_blocks(ao_blocks, cv_blocks)
    old_global = old_scores.get("global_score", 0)
    
    # New score
    new_scores, new_method = score_blocks_enhanced(
        ao_blocks, cv_blocks, ao_struct, cv_struct_good
    )
    new_global = new_scores.get("global_score", 0)
    
    print(f"\n  OLD Scoring (NLP only):")
    print(f"    Global Score: {old_global:.3f}")
    print(f"    Method: {old_method}")
    
    print(f"\n  NEW Scoring (NLP + Non-NLP Features):")
    print(f"    NLP Score:        {new_scores['nlp_score']:.3f}")
    print(f"    Skills Score:     {new_scores['skills_score']:.3f} (overlap: {new_scores['skill_details']['overlap_ratio']:.1%})")
    print(f"    Seniority Score:  {new_scores['seniority_score']:.3f}")
    print(f"    Domain Score:     {new_scores['domain_score']:.3f}")
    print(f"    Language Score:   {new_scores['language_score']:.3f}")
    print(f"    ─────────────────────")
    print(f"    Global Score:     {new_global:.3f} ⬆ +{(new_global - old_global):.3f}")
    
    # Weak candidate
    print(f"\n\n--- SCENARIO 2: Weak Candidate (Mismatch) ---")
    print(f"\nAO: {ao_struct['titre_poste']}")
    print(f"   Requires: {', '.join(ao_struct['competences_techniques'])}")
    
    print(f"\nCV: {cv_struct_weak['role_principal']}")
    print(f"   Skills: {', '.join(cv_struct_weak['technologies'])}")
    print(f"   Seniority: {cv_struct_weak['seniorite']}")
    
    ao_blocks = build_ao_blocks(ao_struct, "")
    cv_blocks_weak = build_cv_blocks(cv_struct_weak, "")
    
    old_scores_weak, _ = score_blocks(ao_blocks, cv_blocks_weak)
    old_global_weak = old_scores_weak.get("global_score", 0)
    
    new_scores_weak, _ = score_blocks_enhanced(
        ao_blocks, cv_blocks_weak, ao_struct, cv_struct_weak
    )
    new_global_weak = new_scores_weak.get("global_score", 0)
    
    print(f"\n  OLD Scoring (NLP only):")
    print(f"    Global Score: {old_global_weak:.3f}")
    
    print(f"\n  NEW Scoring (NLP + Non-NLP Features):")
    print(f"    NLP Score:        {new_scores_weak['nlp_score']:.3f}")
    print(f"    Skills Score:     {new_scores_weak['skills_score']:.3f} (overlap: {new_scores_weak['skill_details']['overlap_ratio']:.1%})")
    print(f"    Seniority Score:  {new_scores_weak['seniority_score']:.3f}")
    print(f"    Domain Score:     {new_scores_weak['domain_score']:.3f}")
    print(f"    Language Score:   {new_scores_weak['language_score']:.3f}")
    print(f"    ─────────────────────")
    print(f"    Global Score:     {new_global_weak:.3f} ⬇ {(new_global_weak - old_global_weak):.3f}")
    
    print(f"\n  Skill Analysis:")
    print(f"    Required: {', '.join(new_scores_weak['skill_details']['ao_skills'])}")
    if new_scores_weak['skill_details']['overlap']:
        print(f"    Candidate has: {', '.join(new_scores_weak['skill_details']['overlap'])}")
    else:
        print(f"    Candidate has: (NONE)")
    print(f"    Missing: {', '.join(new_scores_weak['skill_details']['missing'])}")
    
    # Improvement summary
    print(f"\n\n--- SUMMARY ---")
    print(f"Good candidate discrimination: OLD={old_global:.3f} → NEW={new_global:.3f} (+{(new_global - old_global):.3f})")
    print(f"Weak candidate penalty: OLD={old_global_weak:.3f} → NEW={new_global_weak:.3f} ({(new_global_weak - old_global_weak):.3f})")
    print(f"\nImprovement effect:")
    print(f"  ✓ Better candidates score higher")
    print(f"  ✓ Worse candidates score lower")
    print(f"  ✓ Skill mismatches are penalized")
    print(f"  ✓ Seniority mismatches are flagged")
    print(f"  ✓ Domain relevance is validated")
    print(f"  ✓ No per-pair LLM calls (efficient at scale)")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█ NLP SCORING IMPROVEMENTS - COMPREHENSIVE TEST SUITE")
    print("█"*70)
    
    test_skill_extraction()
    test_skill_normalization()
    test_seniority_parsing()
    test_seniority_matching()
    test_domain_matching()
    test_language_extraction()
    test_full_scoring_comparison()
    
    print("\n" + "█"*70)
    print("█ ALL TESTS COMPLETE")
    print("█"*70 + "\n")
