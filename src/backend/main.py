from retrieval import VectorDBBuilder

def main():
    config_file = '../config/config.yaml'
    builder = VectorDBBuilder(config_file)
    builder.build_all_vector_dbs()

if __name__ == '__main__':
    main()