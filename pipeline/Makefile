# Makefile for Global Food-Genome Pipeline
# Usage: make phase1, make phase2, etc.

.PHONY: all phase0 phase1 phase2 clean help

PYTHON := python3
SCRIPTS_DIR := scripts

# Color output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help:
	@echo "Global Food-Genome Pipeline"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  make phase0  - Validate dataset license registry"
	@echo "  make phase1  - Build ingredient ontology"
	@echo "  make phase2  - Build food chemistry database"
	@echo "  make phase3  - Analyze culinary co-occurrence"
	@echo "  make phase4  - Build bioactivity database"
	@echo "  make phase5  - Map pathways"
	@echo "  make phase6  - Process transcriptomics"
	@echo "  make phase7  - Extract Earth observation features"
	@echo "  make phase8  - Train origin-chemistry model"
	@echo "  make phase9  - Run end-to-end inference"
	@echo "  make clean   - Remove processed data"
	@echo ""

# Phase 0: License Registry
phase0:
	@echo "$(GREEN)Phase 0: Dataset License Registry$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/phase0_dataset_registry/build_license_registry.py

# Phase 1: Ingredient Ontology
phase1:
	@echo "$(GREEN)Phase 1: Building Ingredient Ontology$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/phase1_ingredient_ontology/01_download_usda.py
	$(PYTHON) $(SCRIPTS_DIR)/phase1_ingredient_ontology/02_download_wikidata.py
	$(PYTHON) $(SCRIPTS_DIR)/phase1_ingredient_ontology/03_build_ingredient_master.py

# Phase 2: Food Chemistry
phase2:
	@echo "$(GREEN)Phase 2: Building Food Chemistry Database$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/phase2_food_chemistry/01_download_foodb.py
	# Add more scripts as implemented

# Phase 3: Culinary Matrix
phase3:
	@echo "$(GREEN)Phase 3: Analyzing Culinary Co-occurrence$(NC)"
	# TODO: Implement phase 3 scripts

# Phase 4: Bioactivity
phase4:
	@echo "$(GREEN)Phase 4: Building Bioactivity Database$(NC)"
	# TODO: Implement phase 4 scripts

# Phase 5: Pathways
phase5:
	@echo "$(GREEN)Phase 5: Mapping Pathways$(NC)"
	# TODO: Implement phase 5 scripts

# Phase 6: Transcriptomics
phase6:
	@echo "$(GREEN)Phase 6: Processing Transcriptomics$(NC)"
	# TODO: Implement phase 6 scripts

# Phase 7: Earth Observation
phase7:
	@echo "$(GREEN)Phase 7: Extracting Earth Observation Features$(NC)"
	# TODO: Implement phase 7 scripts

# Phase 8: Origin-Chemistry Model
phase8:
	@echo "$(GREEN)Phase 8: Training Origin-Chemistry Model$(NC)"
	# TODO: Implement phase 8 scripts

# Phase 9: Full Pipeline
phase9:
	@echo "$(GREEN)Phase 9: Running End-to-End Inference$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/phase9_full_pipeline/end_to_end_inference.py

# Clean generated data
clean:
	@echo "$(YELLOW)Cleaning processed data...$(NC)"
	rm -rf data/interim/*
	rm -rf data/processed/*
	@echo "$(GREEN)Clean complete$(NC)"

# Run all phases sequentially
all: phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9
	@echo "$(GREEN)All phases complete!$(NC)"
