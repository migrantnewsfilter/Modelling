################################################################################
#######################KERAS NEURAL NETWORK#####################################
################################################################################

batch_size = 32
nb_epoch = 20
in_out_neurons = news_feeds_df_tfidf.shape[1]
dimof_middle = 100

model = Sequential([
    Dense(batch_size, batch_input_shape=(None, in_out_neurons)),
    Activation('relu'),
    Dropout(0.2),
    Dense(batch_size),
    Activation('relu'),
    Dropout(0.2),
    Dense(in_out_neurons),
    Activation('linear'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

history = model.fit(news_feeds_df_tfidf, np.array(news_feeds_df['label']),
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1
)

, validation_data=(X_test, y_test)


model.1 = Sequential([
    Dense(output_dim=32, input_dim=2145),
    Activation("relu"),
    Dense(10),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

model.fit(news_feeds_df_tfidf, news_feeds_df['label'])

model.fit(news_feeds_df_tfidf, news_feeds_df['label'], nb_epoch=5, batch_size=32)

#Feeding batches to model separately
#model.train_on_batch(X_batch, Y_batch)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
