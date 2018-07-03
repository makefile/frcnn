/*
   Copyright 2015 By ihciah
   https://github.com/ihciah/CNN_forward
   modify by makefile@github
*/
#pragma once

#ifndef __blowfish__
#define __blowfish__

#include <stdint.h>
#include <cstddef>
#include <vector>
#include <string>

class Blowfish {
    public:
        Blowfish(const std::vector<char> &key);
        std::vector<char> Encrypt(const std::vector<char> &src) const;
        std::vector<char> Decrypt(const std::vector<char> &src) const;
        void Encrypt(const char* in_filename, const char* out_filename);
        void Decrypt(const char* in_filename, const char* out_filename);
        std::vector<char> ReadAllBytes(const char* filename);
        void WriteAllBytes(const char* filename, const std::vector<char> &data);
        std::string getRandomTmpFile();

    private:
        void SetKey(const char *key, size_t byte_length);
        void EncryptBlock(uint32_t *left, uint32_t *right) const;
        void DecryptBlock(uint32_t *left, uint32_t *right) const;
        uint32_t Feistel(uint32_t value) const;

    private:
        uint32_t pary_[18];
        uint32_t sbox_[4][256];
};

#endif /* defined(__blowfish__) */
