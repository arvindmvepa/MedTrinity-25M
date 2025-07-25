#!/usr/bin/env python3
"""
Generate summary and insights from clinical evaluation metrics
"""

import json
import numpy as np

def load_metrics():
    """Load the evaluation metrics"""
    with open('clinical_evaluation_metrics.json', 'r') as f:
        return json.load(f)

def generate_summary(metrics):
    """Generate summary insights"""
    
    print("=" * 80)
    print("CLINICAL ANNOTATIONS EVALUATION SUMMARY")
    print("=" * 80)
    
    # Overall performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Overall Accuracy: {metrics['overall_average']:.1%}")
    
    # Task ranking
    print(f"\nTASK PERFORMANCE RANKING:")
    task_scores = [(task, score) for task, score in metrics['task_averages'].items()]
    task_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (task, score) in enumerate(task_scores, 1):
        task_name = task.capitalize()
        if task == 'region':
            task_name += " (Multi-label)"
        else:
            task_name += " (Multi-class)"
        print(f"{i}. {task_name:<25}: {score:.1%}")
    
    # Label type performance
    print(f"\nLABEL TYPE PERFORMANCE:")
    label_averages = {}
    
    for label_type, tasks in metrics['per_label_per_task'].items():
        scores = []
        for task in ['area', 'region', 'shape', 'satellite']:
            if f'{task}_avg' in tasks:
                scores.append(tasks[f'{task}_avg'])
        if scores:
            label_averages[label_type] = np.mean(scores)
    
    # Sort by performance
    sorted_labels = sorted(label_averages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (label, avg) in enumerate(sorted_labels, 1):
        print(f"{i}. {label:<45}: {avg:.1%}")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    # Best and worst performing tasks
    best_task, best_score = task_scores[0]
    worst_task, worst_score = task_scores[-1]
    
    print(f"• Best performing task: {best_task.capitalize()} ({best_score:.1%})")
    print(f"• Worst performing task: {worst_task.capitalize()} ({worst_score:.1%})")
    
    # Performance gaps
    if best_score - worst_score > 0.3:
        print(f"• Large performance gap between best and worst tasks ({best_score - worst_score:.1%})")
    
    # Region performance (multi-label specific)
    region_score = metrics['task_averages']['region']
    if region_score > 0.6:
        print(f"• Good multi-label region prediction performance ({region_score:.1%})")
    elif region_score < 0.5:
        print(f"• Poor multi-label region prediction performance ({region_score:.1%})")
    
    # Shape and satellite performance (often challenging)
    shape_score = metrics['task_averages']['shape']
    satellite_score = metrics['task_averages']['satellite']
    
    if shape_score < 0.3:
        print(f"• Shape prediction is very challenging ({shape_score:.1%} accuracy)")
    if satellite_score < 0.3:
        print(f"• Satellite lesion prediction is very challenging ({satellite_score:.1%} accuracy)")
    
    # Coverage
    coverage_rate = metrics['coverage']['matched_cases'] / metrics['coverage']['total_clinical_cases']
    if coverage_rate == 1.0:
        print(f"• Perfect coverage: All clinical cases matched with predictions")
    else:
        print(f"• Coverage issue: Only {coverage_rate:.1%} of clinical cases matched")

def detailed_analysis(metrics):
    """Provide detailed analysis"""
    
    print(f"\nDETAILED ANALYSIS:")
    print("-" * 80)
    
    # Task-specific analysis
    tasks = ['area', 'region', 'shape', 'satellite']
    task_descriptions = {
        'area': 'Volume/Size assessment',
        'region': 'Anatomical localization (multi-label)',
        'shape': 'Morphological characteristics',
        'satellite': 'Lesion distribution pattern'
    }
    
    for task in tasks:
        score = metrics['task_averages'][task]
        print(f"\n{task.upper()} - {task_descriptions[task]}:")
        print(f"  Overall accuracy: {score:.1%}")
        
        # Per-label breakdown for this task
        label_scores = []
        for label_type, tasks_data in metrics['per_label_per_task'].items():
            if f'{task}_avg' in tasks_data:
                label_score = tasks_data[f'{task}_avg']
                label_scores.append((label_type, label_score))
        
        label_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Best performing label: {label_scores[0][0]} ({label_scores[0][1]:.1%})")
        print(f"  Worst performing label: {label_scores[-1][0]} ({label_scores[-1][1]:.1%})")
        
        # Consistency analysis
        scores_only = [score for _, score in label_scores]
        std_dev = np.std(scores_only)
        if std_dev > 0.2:
            print(f"  High variability across labels (std: {std_dev:.3f})")
        else:
            print(f"  Consistent performance across labels (std: {std_dev:.3f})")

def main():
    """Main function"""
    try:
        metrics = load_metrics()
        generate_summary(metrics)
        detailed_analysis(metrics)
        
    except FileNotFoundError:
        print("Error: clinical_evaluation_metrics.json not found.")
        print("Please run evaluate_clinical_metrics.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
