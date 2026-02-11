import numpy as np

class LogicVectorEncoder:
    def __init__(self):
        # Based on crate::actions::attacks::Mechanic
        self.mechanic_types = [
            "SelfHeal", "SearchToHandByEnergy", "SearchToBenchByName",
            "SearchToHandSupporterCard", "InflictStatusConditions",
            "ChanceStatusAttack", "DamageAllOpponentPokemon",
            "DiscardRandomGlobalEnergy", "DiscardEnergyFromOpponentActive",
            "ExtraDamageIfEx", "SelfDamage", "CoinFlipExtraDamage",
            "CoinFlipExtraDamageOrSelfDamage", "ExtraDamageForEachHeads",
            "CoinFlipNoEffect", "SelfDiscardEnergy", "ExtraDamageIfExtraEnergy",
            "ExtraDamageIfBothHeads", "DirectDamage", "DamageAndTurnEffect",
            "SelfChargeActive", "ManaphyOceanicGift", "PalkiaExDimensionalStorm",
            "MegaBlazikenExMegaBurningAttack", "MoltresExInfernoDance",
            "CelebiExPowerfulBloom", "MagikarpWaterfallEvolution",
            "CoinFlipToBlockAttackNextTurn", "MoveAllEnergyTypeToBench",
            "ChargeBench", "VaporeonHyperWhirlpool", "ConditionalBenchDamage",
            "ExtraDamageForEachHeadsWithStatus", "DamageAndMultipleCardEffects",
            "DamageReducedBySelfDamage", "ExtraDamagePerTrainerInOpponentDeck",
            "ExtraDamageIfCardInDiscard", "DamageAndCardEffect",
            "SelfDiscardAllEnergy", "SelfDiscardRandomEnergy",
            "AlsoBenchDamage", "AlsoChoiceBenchDamage", "ExtraDamageIfHurt",
            "DamageEqualToSelfDamage", "ExtraDamageEqualToSelfDamage",
            "ExtraDamageIfKnockedOutLastTurn", "BenchCountDamage",
            "EvolutionBenchCountDamage", "ExtraDamagePerEnergy",
            "ExtraDamagePerRetreatCost", "DamagePerEnergyAll",
            "ExtraDamagePerSpecificEnergy", "ExtraDamageIfToolAttached",
            "RecoilIfKo", "ShuffleOpponentActiveIntoDeck",
            "BlockBasicAttack", "SwitchSelfWithBench"
        ]
        self.type_to_idx = {name: i for i, name in enumerate(self.mechanic_types)}
        # [One-hot Type] + [amount, damage, extra_damage, self_damage, num_coins, duration, probability]
        self.vector_size = len(self.mechanic_types) + 8

    def encode(self, attack_info: dict) -> np.ndarray:
        vec = np.zeros(self.vector_size, dtype=np.float32)
        if not attack_info:
            return vec

        # The serialized format from Rust (untagged or adjacently tagged)
        # With #[derive(Serialize)], it looks like {"SelfHeal": {"amount": 30}}
        # or for unit variant just "DiscardRandomGlobalEnergy"
        
        if isinstance(attack_info, str):
            m_type = attack_info
            params = {}
        elif isinstance(attack_info, dict):
            # Adjacently tagged (default for enums in serde if not specified)
            # Actually, standard is "Externally tagged" for enums: {"Variant": {params}}
            m_type = list(attack_info.keys())[0] if attack_info else ""
            params = attack_info[m_type] if m_type else {}
        else:
            return vec

        if m_type in self.type_to_idx:
            idx = self.type_to_idx[m_type]
            vec[idx] = 1.0
        
        offset = len(self.mechanic_types)
        
        # Mapping common parameters to fixed slots
        if isinstance(params, dict):
            # amount / damage (slot 0)
            val = params.get("amount") or params.get("damage")
            if val is not None:
                vec[offset + 0] = float(val) / 300.0
            
            # extra_damage (slot 1)
            val = params.get("extra_damage")
            if val is not None:
                vec[offset + 1] = float(val) / 300.0
                
            # self_damage (slot 2)
            val = params.get("self_damage")
            if val is not None:
                vec[offset + 2] = float(val) / 300.0
                
            # num_coins (slot 3)
            val = params.get("num_coins")
            if val is not None:
                vec[offset + 3] = float(val) / 10.0
                
            # duration (slot 4)
            val = params.get("duration")
            if val is not None:
                vec[offset + 4] = float(val) / 5.0
                
            # probability (slot 5)
            val = params.get("probability")
            if val is not None:
                vec[offset + 5] = float(val)
            elif "Choice" in m_type or "Chance" in m_type or "CoinFlip" in m_type:
                vec[offset + 5] = 0.5 # Default for coin flip if not specified
                
            # bench_only (slot 6)
            if params.get("bench_only"):
                 vec[offset + 6] = 1.0

        return vec

class TrainerVectorEncoder:
    def __init__(self):
        self.mechanic_types = [
            "EvolutionAcceleration", "Heal", "Draw", "EnergyAttachment", 
            "Switch", "Search", "DamageBoost", "RetreatCostReduction",
            "ShuffleHandInDraw", "AttachTool", "PlaceFossil", "MultiEffect"
        ]
        self.target_scopes = [
            "SelfActive", "SelfBench", "SelfBoard", 
            "OpponentActive", "OpponentBench", "OpponentBoard"
        ]
        
        self.type_to_idx = {name: i for i, name in enumerate(self.mechanic_types)}
        self.scope_to_idx = {name: i for i, name in enumerate(self.target_scopes)}
        
        # [One-hot Type] + [One-hot Scope] + [amount, stages_to_skip, flags(cure, forced, shuffle, ex)]
        self.vector_size = len(self.mechanic_types) + len(self.target_scopes) + 6

    def encode(self, mechanic_info: dict) -> np.ndarray:
        vec = np.zeros(self.vector_size, dtype=np.float32)
        if not mechanic_info:
            return vec

        # Format from Rust #[serde(tag = "type", content = "params")]
        # {"type": "Heal", "params": {"amount": 20, "cure_status": false}}
        m_type = mechanic_info.get("type", "")
        params = mechanic_info.get("params", {})
        
        if m_type == "MultiEffect":
            # Combine effects (max pooling)
            effects = params.get("effects", [])
            if effects:
                all_vecs = [self.encode(e) for e in effects]
                return np.max(all_vecs, axis=0)
            return vec

        if m_type in self.type_to_idx:
            idx = self.type_to_idx[m_type]
            vec[idx] = 1.0
            
        offset_scope = len(self.mechanic_types)
        offset_params = offset_scope + len(self.target_scopes)

        if isinstance(params, dict):
            # Target Scope
            scope = params.get("target_scope") or params.get("target")
            if scope in self.scope_to_idx:
                vec[offset_scope + self.scope_to_idx[scope]] = 1.0

            # amount (slot 0)
            if "amount" in params:
                vec[offset_params + 0] = float(params["amount"]) / 100.0
                
            # stages_to_skip (slot 1)
            if "stages_to_skip" in params:
                vec[offset_params + 1] = float(params["stages_to_skip"])
                
            # flags
            if params.get("cure_status"):
                vec[offset_params + 2] = 1.0
            if params.get("forced"):
                vec[offset_params + 3] = 1.0
            if params.get("shuffle_hand_first"):
                vec[offset_params + 4] = 1.0
            if params.get("against_ex"):
                vec[offset_params + 5] = 1.0
                
        return vec

class AbilityVectorEncoder:
    def __init__(self):
        self.mechanic_types = [
            "HealAllYourPokemon", "DamageOneOpponentPokemon", "SwitchActiveTypedWithBench",
            "ReduceDamageFromAttacks", "StartTurnRandomPokemonToHand", "PreventFirstAttack",
            "ElectromagneticWall"
        ]
        self.energy_types = [
            "Grass", "Fire", "Water", "Lightning", "Psychic",
            "Fighting", "Darkness", "Metal", "Dragon", "Colorless"
        ]
        self.type_to_idx = {name: i for i, name in enumerate(self.mechanic_types)}
        self.energy_to_idx = {name: i for i, name in enumerate(self.energy_types)}
        
        # [One-hot Type] + [amount] + [One-hot EnergyType]
        self.vector_size = len(self.mechanic_types) + 1 + len(self.energy_types)

    def encode(self, ability_info: dict) -> np.ndarray:
        vec = np.zeros(self.vector_size, dtype=np.float32)
        if not ability_info:
            return vec

        if isinstance(ability_info, str):
            m_type = ability_info
            params = {}
        elif isinstance(ability_info, dict):
            m_type = list(ability_info.keys())[0] if ability_info else ""
            params = ability_info[m_type] if m_type else {}
        else:
            return vec

        if m_type in self.type_to_idx:
            idx = self.type_to_idx[m_type]
            vec[idx] = 1.0
        
        offset_amount = len(self.mechanic_types)
        offset_energy = offset_amount + 1

        if isinstance(params, dict):
            # amount (slot 0)
            if "amount" in params:
                vec[offset_amount] = float(params["amount"]) / 100.0
            
            # energy_type (one-hot)
            energy = params.get("energy_type")
            if energy in self.energy_to_idx:
                vec[offset_energy + self.energy_to_idx[energy]] = 1.0

        return vec

