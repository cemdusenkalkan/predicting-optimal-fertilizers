#!/usr/bin/env bash
set -e

echo "🚀 Running Competitive Fertilizer Pipeline Tests"
echo "================================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models submissions data/processed experiments

# Test all modules sequentially
echo ""
echo "🔄 Testing DataLoader..."
python test_data_loader.py

echo ""
echo "🔧 Testing FeatureEngineering..."
python test_feature_engineering.py

echo ""
echo "🧗 Testing HillClimbing..."
python test_hill_climbing.py

echo ""
echo "🚀 Testing AutoGluonTrainer..."
python test_autogluon_trainer.py

echo ""
echo "📈 Testing MetaStacking..."
python test_meta_stacking.py

echo ""
echo "🎯 Testing RankingModels..."
python test_ranking_models.py

echo ""
echo "✅ All module tests completed successfully!"
echo ""
echo "🏆 To run the full competitive pipeline:"
echo "   python run_competition.py --mode meta-stacking"
echo ""
echo "📊 To evaluate model performance:"
echo "   python run_competition.py --mode evaluate" 