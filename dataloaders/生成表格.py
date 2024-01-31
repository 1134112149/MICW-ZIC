import pandas as pd

# List of words to be inserted into an Excel file
words_list = ['tarantula', 'centipede', 'goose', 'koala', 'jellyfish', 'snail', 'slug', 'American_lobster',
              'spiny_lobster', 'black_stork', 'king_penguin', 'albatross', 'dugong', 'Chihuahua', 'Yorkshire_terrier',
              'golden_retriever', 'Labrador_retriever', 'German_shepherd', 'standard_poodle', 'tabby', 'Persian_cat',
              'Egyptian_cat', 'cougar', 'lion', 'brown_bear', 'ladybug', 'fly', 'bee', 'grasshopper', 'walking_stick',
              'cockroach', 'mantis', 'dragonfly', 'sulphur_butterfly', 'sea_cucumber', 'guinea_pig', 'hog', 'ox',
              'bison', 'bighorn', 'gazelle', 'Arabian_camel', 'orangutan', 'chimpanzee', 'baboon', 'African_elephant',
              'abacus', 'academic_gown', 'altar', 'apron', 'backpack', 'bannister', 'barbershop', 'barrel', 'basketball',
              'bathtub', 'beach_wagon', 'beacon', 'beer_bottle', 'bikini', 'birdhouse', 'bow_tie', 'brass', 'broom',
              'bucket', 'butcher_shop', 'candle', 'cannon', 'cardigan', 'cash_machine', 'CD_player', 'chain', 'chest',
              'Christmas_stocking', 'cliff_dwelling', 'computer_keyboard', 'confectionery', 'convertible', 'crane',
              'dam', 'desk', 'dining_table', 'drumstick', 'dumbbel', 'flagpole', 'fountain', 'freight_car', 'frying_pan',
              'fur_coat', 'gasmask', 'go-kart', 'hourglass', 'iPod', 'jinrikisha', 'kimono', 'lampshade', 'lawn_mower',
              'lifeboat', 'limousine', 'magnetic_compass', 'maypole', 'military_uniform', 'miniskirt', 'moving_van',
              'nail', 'obelisk', 'oboe', 'organ', 'parking_meter', 'pay-phone', 'picket_fence', 'pill_bottle', 'plunger',
              'pole', 'pop_bottle', "potter's_wheel", 'projectile', 'punching_bag', 'reel', 'refrigerator',
              'remote_control', 'rocking_chair', 'rugby_ball', 'school_bus', 'scoreboard', 'snorkel', 'sock', 'sombrero',
              'space_heater', 'sports_car', 'steel_arch_bridge', 'stopwatch', 'sunglasses', 'suspension_bridge',
              'swimming_trunks', 'syringe', 'teapot', 'teddy', 'thatch', 'torch', 'tractor', 'triumphal_arch',
              'trolleybus', 'umbrella', 'vestment', 'viaduct', 'water_jug', 'water_tower', 'wok', 'wooden_spoon',
              'comic_book', 'plate', 'guacamole', 'ice_cream', 'ice_lolly', 'pretzel', 'mashed_potato', 'cauliflower',
              'bell_pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat_loaf', 'pizza', 'potpie',
              'alp', 'cliff', 'coral_reef', 'seashore', 'acorn']

# Creating a DataFrame with the words in a single column
df = pd.DataFrame(words_list, columns=['Words'])

# Saving the DataFrame to an Excel file
excel_file_path = 'E:\wsnbb\y\LMC-main\dataloaders\words_list.xlsx'
df.to_excel(excel_file_path, index=False)

excel_file_path
